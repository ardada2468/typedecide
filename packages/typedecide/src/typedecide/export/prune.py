"""Vocabulary pruning: slicing the unembedding down to the answer letters.

A typed readout never generates text. It reads the logits at one position and
compares at most ``len(LETTERS)`` of them. Every other row of the unembedding
matrix is computed, copied out of the GPU, and thrown away on every single call.

So we slice the matrix. For ``Qwen/Qwen3-0.6B`` the table is 151936 x 1024 =
155,582,464 parameters -- 26.1% of the model's 596,049,920 -- and the returned
logits tensor drops from ``[1, seq, 151936]`` to ``[1, seq, K]``. In a browser
that tensor is copied out of GPU memory on every call, which is the part that
actually hurts. ``docs/SPEED.md`` carries the full arithmetic and is explicit
about which numbers were measured and which were derived.

What a pruned model can no longer do
------------------------------------
A pruned model **cannot generate text.** There is no row for any token other
than the kept ones, so sampling, beam search, perplexity, and every other
generative use are gone permanently. Re-export from the unpruned checkpoint if
you need them.

A pruned model also **cannot report what share of the total vocabulary mass
landed on the answer letters.** The softmax denominator over the full
vocabulary no longer exists. The probabilities you get are a softmax over the K
kept rows and nothing else, which is exactly what the readout uses -- but it
means the diagnostic "the model actually wanted to say something that is not an
option" is no longer available from a pruned model. Keep an unpruned copy if
that number matters to you.

The correctness trap
--------------------
Qwen3 at small sizes sets ``config.tie_word_embeddings = True``: ``lm_head.weight``
and ``model.embed_tokens.weight`` are *the same tensor object*. Slicing it in
place would leave the input embedding with K rows, so every input token id above
K would index out of bounds -- or, worse, silently read the wrong row. So we
untie first: build a fresh K-row tensor, rebind it to the head only, clear
``config.tie_word_embeddings`` and ``model._tied_weights_keys`` (otherwise
``save_pretrained`` drops the head as a "tied" duplicate), and then verify the
input embedding is unchanged before returning. ``prune_lm_head`` refuses to
return a model that failed that check.

Note that ``config.vocab_size`` is deliberately **not** changed. The input
embedding still needs a row for every token id in the prompt; only the output
side shrank.
"""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExportError
from ..prompt import Tokenizer, answer_slots, full_prompt
from ..schema import LETTERS, MAX_OPTIONS, MIN_OPTIONS, Criterion, Option

log = logging.getLogger("typedecide.export.prune")

SLOT_MAP_FILENAME = "slot_map.json"
SLOT_MAP_VERSION = 1

# A neutral probe. Only the boundary after "Answer:" decides the answer-letter
# ids, so the body just has to be ordinary prose that tokenizes without drama.
PROBE_STATE = "The applicant submitted the form on Tuesday and attached two documents."
PROBE_QUESTION = "Which option applies?"


# ------------------------------------------------------------------ slot maps


@dataclass(frozen=True, slots=True)
class SlotMap:
    """Which original vocabulary id each row of the pruned head came from.

    After pruning, the readout indexes rows ``0..K-1``. It must not index
    vocabulary ids any more -- those rows are gone. Anything that consumes a
    pruned model (the browser runtime, an evaluation harness, a debugging
    script) needs this file to know that row 2 used to be token 33.

    ``keep[row] -> original token id`` and ``token_to_row[token id] -> row`` are
    the two directions, and both are written to ``slot_map.json`` so the
    consumer does not have to invert anything.
    """

    model_id: str
    separator: str | None
    letters: str
    keep: tuple[int, ...]
    original_vocab_size: int | None = None

    @property
    def rows(self) -> int:
        """K -- the number of rows the pruned unembedding has."""
        return len(self.keep)

    @property
    def token_to_row(self) -> dict[int, int]:
        return {token: row for row, token in enumerate(self.keep)}

    def row_of(self, token_id: int) -> int:
        """The pruned row holding `token_id`'s logit."""
        try:
            return self.keep.index(token_id)
        except ValueError:
            raise ExportError(
                f"Token id {token_id} is not in this slot map, so the pruned model has no "
                f"row for it. The map covers {list(self.keep)} (letters {self.letters!r}); "
                "a token outside that set was dropped by pruning and cannot be recovered."
            ) from None

    def token_of(self, row: int) -> int:
        """The original vocabulary id that `row` of the pruned head came from."""
        if not 0 <= row < len(self.keep):
            raise ExportError(
                f"Row {row} is outside the pruned head, which has {len(self.keep)} rows "
                f"(0..{len(self.keep) - 1})."
            )
        return self.keep[row]

    def letter_of(self, row: int) -> str:
        """The answer letter `row` scores."""
        if not 0 <= row < len(self.letters):
            raise ExportError(
                f"Row {row} is outside the {len(self.letters)} letters this map covers."
            )
        return self.letters[row]

    def to_dict(self) -> dict[str, Any]:
        """The on-disk shape. JSON object keys are strings, so `token_to_row` is
        keyed by the decimal token id -- consumers must parse them back to ints."""
        return {
            "version": SLOT_MAP_VERSION,
            "model_id": self.model_id,
            "separator": self.separator,
            "letters": self.letters,
            "rows": self.rows,
            "original_vocab_size": self.original_vocab_size,
            "keep": list(self.keep),
            "token_to_row": {str(token): row for row, token in enumerate(self.keep)},
            "can_generate_text": False,
            "note": (
                "Row index, not vocabulary id. A model pruned with this map cannot "
                "generate text and cannot report the share of total vocabulary mass on "
                "the answer letters."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SlotMap:
        if not isinstance(payload, dict):
            raise ExportError(
                f"A slot map must be a JSON object, got {type(payload).__name__}."
            )
        version = payload.get("version", SLOT_MAP_VERSION)
        if version != SLOT_MAP_VERSION:
            raise ExportError(
                f"This slot map is version {version!r} but this build of typedecide writes "
                f"and reads version {SLOT_MAP_VERSION}. Re-export the model."
            )
        missing = {"model_id", "letters", "keep"} - payload.keys()
        if missing:
            raise ExportError(
                f"A slot map needs {sorted({'model_id', 'letters', 'keep'})}; "
                f"this one is missing {sorted(missing)}."
            )
        keep = payload["keep"]
        if not isinstance(keep, list):
            raise ExportError("A slot map's 'keep' must be a list of token ids in row order.")
        slot_map = build_slot_map(
            keep,
            model_id=str(payload["model_id"]),
            separator=None if payload.get("separator") is None else str(payload["separator"]),
            letters=str(payload["letters"]),
            original_vocab_size=payload.get("original_vocab_size"),
        )
        stored = payload.get("token_to_row")
        if isinstance(stored, dict):
            expected = {str(token): row for row, token in enumerate(slot_map.keep)}
            try:
                actual: dict[str, int] | None = {str(k): int(v) for k, v in stored.items()}
            except (TypeError, ValueError):
                actual = None  # a row that is not a number cannot agree with anything
            if actual != expected:
                raise ExportError(
                    "This slot map's 'token_to_row' disagrees with its 'keep' order. One of "
                    "the two was edited by hand; the file cannot be trusted. Re-export."
                )
        return slot_map


def build_slot_map(
    keep: Sequence[int],
    *,
    model_id: str,
    separator: str | None,
    letters: str | None = None,
    original_vocab_size: int | None = None,
) -> SlotMap:
    """Validate `keep` and record what it means.

    `keep` is ordered: row *i* of the pruned head scores ``letters[i]``. Order is
    preserved exactly as given, never sorted -- sorting would silently permute
    the answer letters, which is the kind of bug that shows up as a model that
    is confidently wrong rather than as a crash.
    """
    if not model_id or not model_id.strip():
        raise ExportError(
            "A slot map must name the model it was built for, so a consumer can refuse a "
            "mismatched pair of weights and map."
        )
    index = validate_keep(keep, vocab_size=original_vocab_size)
    if letters is None:
        if len(index) > len(LETTERS):
            raise ExportError(
                f"Asked to keep {len(index)} rows but only {len(LETTERS)} answer letters "
                f"exist ({LETTERS}). Pass `letters` explicitly if these rows are not "
                "answer letters."
            )
        letters = LETTERS[: len(index)]
    if len(letters) != len(index):
        raise ExportError(
            f"{len(index)} kept rows but {len(letters)} letters ({letters!r}). Each row of "
            "the pruned head scores exactly one letter, so the two must be the same length."
        )
    if len(index) > MAX_OPTIONS:
        log.warning(
            "Keeping %d rows, more than the %d answer letters this library renders. That is "
            "allowed, but nothing in typedecide will read the rows past %d.",
            len(index),
            MAX_OPTIONS,
            MAX_OPTIONS - 1,
        )
    return SlotMap(
        model_id=model_id,
        separator=separator,
        letters=letters,
        keep=tuple(index),
        original_vocab_size=None if original_vocab_size is None else int(original_vocab_size),
    )


def validate_keep(keep: Sequence[int], *, vocab_size: int | None = None) -> list[int]:
    """The kept token ids as plain ints, in the order given.

    Raises `ExportError` for an empty set, a non-integer, a negative id, an id at
    or beyond `vocab_size`, or a duplicate. A duplicate is not a harmless quirk:
    two letters would share one row, the readout would tie forever, and nothing
    downstream would notice.
    """
    try:
        index = [_as_index(token) for token in keep]
    except TypeError as error:
        raise ExportError(
            f"Kept token ids must be integers; {error}. They are vocabulary ids from "
            "`answer_letter_token_ids`, not letters or strings."
        ) from error

    if not index:
        raise ExportError(
            "Pruning needs at least one row to keep. An empty `keep` would leave the model "
            "with no output at all."
        )
    if len(index) < MIN_OPTIONS:
        log.warning(
            "Pruning to %d row(s); every criterion this library permits has at least %d "
            "options, so this head cannot answer one.",
            len(index),
            MIN_OPTIONS,
        )

    seen: dict[int, int] = {}
    for position, token in enumerate(index):
        if token < 0:
            raise ExportError(
                f"Token id {token} at position {position} is negative. Vocabulary ids start "
                "at 0; a negative id usually means a tokenizer probe failed and returned a "
                "sentinel."
            )
        if vocab_size is not None and token >= vocab_size:
            raise ExportError(
                f"Token id {token} at position {position} is outside the model's vocabulary "
                f"of {vocab_size} (valid ids are 0..{vocab_size - 1}). The token ids and the "
                "model do not come from the same tokenizer."
            )
        if token in seen:
            detail = ""
            if position < len(LETTERS):
                detail = (
                    f" Letters {LETTERS[seen[token]]!r} and {LETTERS[position]!r} would share "
                    "one row of the pruned head and could never be told apart."
                )
            raise ExportError(
                f"Token id {token} appears twice in `keep`, at positions {seen[token]} and "
                f"{position}. Each kept row must come from a distinct vocabulary id.{detail}"
            )
        seen[token] = position
    return index


def write_slot_map(slot_map: SlotMap, destination: Path | str) -> Path:
    """Write `slot_map.json`. `destination` may be the file or the directory holding it."""
    path = Path(destination)
    if path.suffix.lower() != ".json":
        path = path / SLOT_MAP_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slot_map.to_dict(), indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows, letters %s)", path, slot_map.rows, slot_map.letters)
    return path


def read_slot_map(source: Path | str) -> SlotMap:
    """Read `slot_map.json`. `source` may be the file or the directory holding it."""
    path = Path(source)
    if path.is_dir():
        path = path / SLOT_MAP_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExportError(
            f"No slot map at {path}. A pruned model is unreadable without one, because its "
            "rows are 0..K-1 rather than vocabulary ids. Re-run the export."
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExportError(f"{path} is not valid JSON: {error}.") from error
    except OSError as error:
        raise ExportError(f"Could not read the slot map at {path}: {error}.") from error
    return SlotMap.from_dict(payload)


# ------------------------------------------------------- finding the letter ids


def answer_letter_slots(tokenizer: Tokenizer, n_options: int) -> tuple[list[int], str]:
    """The answer-letter token ids and the separator that produced them.

    Probes at the real prompt boundary, because whether the natural continuation
    is ``" A"`` or ``"A"`` depends on the character before it. Takes a tokenizer
    object so tests can pass a fake one instead of downloading a real model.
    """
    if not MIN_OPTIONS <= n_options <= MAX_OPTIONS:
        raise ExportError(
            f"n_options={n_options} is outside the {MIN_OPTIONS}..{MAX_OPTIONS} options this "
            f"library renders letters for ({LETTERS})."
        )
    criterion = Criterion(
        key="probe",
        question=PROBE_QUESTION,
        options=tuple(
            Option(id=f"opt{i}", description=f"Option {LETTERS[i]}") for i in range(n_options)
        ),
    )
    return answer_slots(tokenizer, full_prompt(PROBE_STATE, criterion), n_options)


def answer_letter_token_ids(model_id: str, n_options: int) -> list[int]:
    """The vocabulary ids of `A`..  for `model_id`, in letter order.

    This is the `keep` list for `prune_lm_head`. Loading the tokenizer needs
    `transformers`.
    """
    transformers = _require("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as error:  # noqa: BLE001 - hub/network/config failures are all the same here
        raise ExportError(
            f"Could not load the tokenizer for {model_id!r}: {error}. The answer-letter ids "
            "come from the tokenizer, so the export cannot continue without it."
        ) from error
    slots, separator = answer_letter_slots(tokenizer, n_options)
    log.info(
        "answer letters for %s: %s (separator %r)", model_id, slots[:n_options], separator
    )
    return slots


# ------------------------------------------------------------------- the slice


def select_rows(weight: Any, keep: Sequence[int]) -> Any:
    """Rows `keep` of `weight`, in that order, as a fresh object.

    Works on a torch tensor and on plain nested lists, which is what lets the
    untie logic be tested without torch installed. The result is always a copy:
    a view onto a tied embedding would defeat the whole point of untying.
    """
    index = [int(i) for i in keep]
    if hasattr(weight, "detach") and hasattr(weight, "index_select"):
        # torch: advanced indexing already copies, clone() makes that a promise.
        return weight.detach()[index].clone()
    return [list(weight[i]) for i in index]


def _select_entries(vector: Any, keep: Sequence[int]) -> Any:
    """`select_rows` for a 1-D thing (a bias)."""
    index = [int(i) for i in keep]
    if hasattr(vector, "detach") and hasattr(vector, "index_select"):
        return vector.detach()[index].clone()
    return [vector[i] for i in index]


def prune_lm_head(model: Any, keep: Sequence[int]) -> Any:
    """Replace the unembedding with the `keep` rows of itself, and return `model`.

    After this the model emits ``[batch, seq, len(keep)]`` logits. Row *i* holds
    what used to be the logit of token ``keep[i]``; pair the model with the
    `SlotMap` that `build_slot_map` makes from the same `keep`, because nothing
    in the weights records it.

    **The pruned model cannot generate text.** Rows for every other token are
    gone. It also **cannot report the share of total vocabulary mass** that landed
    on the answer letters -- the full-vocabulary softmax denominator no longer
    exists, so probabilities are a softmax over the K kept rows only.

    Tied embeddings are handled by untying first. Qwen3 ties at 0.6B, 1.7B and 4B, so
    ``lm_head.weight is embed_tokens.weight``; slicing it in place would corrupt
    the *input* embedding and every prompt token id above K would read the wrong
    row or fail. We rebind the head to a fresh tensor, clear
    ``config.tie_word_embeddings`` and ``model._tied_weights_keys`` so
    ``save_pretrained`` does not drop the new head as a tied duplicate, and then
    check the input embedding still has its original rows and values. If it does
    not, we raise rather than hand back a quietly broken model.

    ``config.vocab_size`` is left alone on purpose: the input side still needs a
    row per token id.
    """
    head = _output_embeddings(model)
    inputs = _input_embeddings(model)
    head_rows = _row_count(head.weight)
    index = validate_keep(keep, vocab_size=head_rows)

    config = getattr(model, "config", None)
    declared_tied = bool(getattr(config, "tie_word_embeddings", False))
    shared_storage = _shares_storage(head.weight, None if inputs is None else inputs.weight)
    if declared_tied and inputs is not None and not shared_storage:
        log.warning(
            "config.tie_word_embeddings is True but the head and the input embedding are "
            "separate tensors. Untying anyway; the config was stale."
        )

    input_rows_before = None if inputs is None else _row_count(inputs.weight)
    input_fingerprint_before = None if inputs is None else _row_fingerprint(inputs.weight, index[0])

    # Rebind rather than mutate. Mutating would reach through to the input
    # embedding whenever the two are tied, which is the whole trap.
    head.weight = _as_parameter(select_rows(head.weight, index))
    if getattr(head, "bias", None) is not None:
        head.bias = _as_parameter(_select_entries(head.bias, index))
    if hasattr(head, "out_features"):
        head.out_features = len(index)

    if config is not None:
        config.tie_word_embeddings = False
    if getattr(model, "_tied_weights_keys", None):
        # Left set, save_pretrained treats lm_head.weight as a duplicate of the
        # input embedding and omits it, and the reloaded model has no head.
        model._tied_weights_keys = []

    _verify_untied(
        model,
        head,
        inputs,
        index,
        input_rows_before=input_rows_before,
        input_fingerprint_before=input_fingerprint_before,
        was_tied=declared_tied or shared_storage,
    )

    log.info(
        "pruned the unembedding from %d rows to %d (%s), input embedding left at %s rows",
        head_rows,
        len(index),
        "untied first" if (declared_tied or shared_storage) else "already untied",
        input_rows_before,
    )
    return model


def _verify_untied(
    model: Any,
    head: Any,
    inputs: Any,
    index: Sequence[int],
    *,
    input_rows_before: int | None,
    input_fingerprint_before: tuple[float, ...] | None,
    was_tied: bool,
) -> None:
    """Refuse to return a model whose input embedding was damaged by the slice."""
    if _row_count(head.weight) != len(index):
        raise ExportError(
            f"The pruned head has {_row_count(head.weight)} rows but {len(index)} were kept. "
            "The slice did not take; do not export this model."
        )
    if inputs is None:
        return
    if head.weight is inputs.weight or _shares_storage(head.weight, inputs.weight):
        raise ExportError(
            "The unembedding is still sharing storage with the input embedding after "
            "pruning, so the input embedding now has too few rows and every prompt token "
            "above the kept set would read the wrong row. Untying failed on this model "
            "class; export from an untied checkpoint instead."
        )
    now_rows = _row_count(inputs.weight)
    if now_rows != input_rows_before:
        raise ExportError(
            f"Pruning the unembedding changed the input embedding from {input_rows_before} "
            f"to {now_rows} rows. The two were tied and the slice reached through. Do not "
            "export this model."
        )
    now_fingerprint = _row_fingerprint(inputs.weight, index[0])
    if now_fingerprint != input_fingerprint_before:
        raise ExportError(
            "Pruning changed the values of the input embedding. The two were tied and the "
            "slice reached through. Do not export this model."
        )
    if was_tied and bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        raise ExportError(
            "config.tie_word_embeddings is still True after pruning, so loading this model "
            "back would re-tie the K-row head onto the input embedding and destroy it."
        )


# ------------------------------------------------------------------ arithmetic


def lm_head_savings(
    vocab_size: int,
    hidden_size: int,
    kept_rows: int,
    *,
    tied: bool = False,
    bits_per_weight: float = 4.5,
    logit_bytes: int = 4,
) -> dict[str, float]:
    """The arithmetic behind every size and speed claim in `docs/SPEED.md`.

    `bits_per_weight` defaults to 4.5: 4-bit weights plus one fp16 scale per
    block of 32, which is what `quantize(mode="q4f16")` produces.

    When `tied` is true, `weight_bytes_removed` is **zero**. That is not a bug
    and it is the single most misunderstood thing about this optimisation: a
    tied model has one table serving both the input embedding and the
    unembedding, the input side still needs every row, so the table cannot leave
    the file. The win on a tied model is the matmul and the logits tensor, not
    the download.
    """
    for name, value in (("vocab_size", vocab_size), ("hidden_size", hidden_size)):
        if value <= 0:
            raise ExportError(f"{name} must be positive, got {value}.")
    if not 0 < kept_rows <= vocab_size:
        raise ExportError(
            f"kept_rows must be between 1 and vocab_size ({vocab_size}), got {kept_rows}."
        )
    head_parameters = vocab_size * hidden_size
    kept_parameters = kept_rows * hidden_size
    removed = head_parameters - kept_parameters
    return {
        "head_parameters": float(head_parameters),
        "kept_parameters": float(kept_parameters),
        "removed_parameters": float(removed),
        "weight_bytes_removed": 0.0 if tied else removed * bits_per_weight / 8.0,
        "logit_bytes_per_position_before": float(vocab_size * logit_bytes),
        "logit_bytes_per_position_after": float(kept_rows * logit_bytes),
        "matmul_flops_per_position_before": float(2 * hidden_size * vocab_size),
        "matmul_flops_per_position_after": float(2 * hidden_size * kept_rows),
    }


# ----------------------------------------------------------------- odds and ends


def _as_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{value!r} is a bool, not a token id")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{value!r} is not an integer ({error})") from error


def _require(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ExportError(
            f"{module!r} is needed here but is not installed. Install the exporter's "
            "dependencies with: pip install 'typedecide[export]' 'typedecide[train]'"
        ) from error


def _output_embeddings(model: Any) -> Any:
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        raise ExportError(
            "This model exposes no output embedding to prune. `prune_lm_head` needs a "
            "causal-LM head (`get_output_embeddings()` or `.lm_head`); an encoder or a "
            "sequence-classification head has nothing to slice."
        )
    return head


def _input_embeddings(model: Any) -> Any:
    getter = getattr(model, "get_input_embeddings", None)
    try:
        inputs = getter() if callable(getter) else None
    except (AttributeError, NotImplementedError):
        inputs = None
    if inputs is None:
        inner = getattr(model, "model", None)
        inputs = getattr(inner, "embed_tokens", None)
    if inputs is not None and not hasattr(inputs, "weight"):
        return None
    return inputs


def _row_count(weight: Any) -> int:
    shape = getattr(weight, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(weight)


def _row_fingerprint(weight: Any, row: int, width: int = 8) -> tuple[float, ...]:
    """Enough of one row to notice if the slice reached through to it."""
    values = weight[row]
    return tuple(float(v) for v in list(values)[:width])


def _shares_storage(left: Any, right: Any) -> bool:
    if right is None:
        return False
    if left is right:
        return True
    left_ptr = getattr(left, "data_ptr", None)
    right_ptr = getattr(right, "data_ptr", None)
    if callable(left_ptr) and callable(right_ptr):
        return bool(left_ptr() == right_ptr())
    return False


def _as_parameter(tensor: Any) -> Any:
    """Wrap a torch tensor back into a Parameter; leave a plain list alone."""
    if not hasattr(tensor, "detach"):
        return tensor
    try:
        import torch
    except ImportError:  # pragma: no cover - a tensor without torch is not a thing
        return tensor
    return torch.nn.Parameter(tensor, requires_grad=False)
