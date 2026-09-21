"""Turning `Decision`s into the one supervised token a typed readout ever reads.

The invariant this whole module exists to hold:

    input_ids = encode(full_prompt(state, criterion)) + [answer_token]
    labels    = [-100] * len(encode(full_prompt(state, criterion))) + [answer_token]
    answer_token = prompt.answer_slots(tokenizer, prompt, n_options)[0][answer_index]

Every token except the last is masked. The browser never samples a sentence: it reads
the logits at the position right after "Answer:" and softmaxes over that criterion's
answer letters, so that position is the only one worth a gradient.

`build_example` is a pure function over the `prompt.Tokenizer` protocol -- `.encode` and
nothing else -- so the invariant is testable against a stub with no torch and no model
download. `DecisionDataset` is the thin indexable wrapper the trainer hands to
`transformers.Trainer`; it deliberately does not subclass `torch.utils.data.Dataset`,
because that base class contributes no behaviour and subclassing it would drag torch
into import time for the sake of a name.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from ..errors import PromptError, SchemaError, TrainingError
from ..prompt import INSTRUCTION, Tokenizer, answer_slots, full_prompt, head_text, tail_text
from ..schema import Decision

log = logging.getLogger("typedecide.training.dataset")

IGNORE_INDEX = -100

# The fixed opening of every prompt: instruction, then the "Evidence:" header. Kept
# when a long state has to be truncated, because a model that never sees the
# instruction during training is being trained on a different task than it is asked.
PREAMBLE = f"{INSTRUCTION}\n\nEvidence:\n"


# --------------------------------------------------------------------- example


def build_example(
    tokenizer: Tokenizer, decision: Decision, max_length: int
) -> dict[str, list[int]]:
    """One training example: the whole prompt masked out, plus one supervised letter.

    Returns `input_ids`, `attention_mask` and `labels`, all the same length and all at
    most `max_length` long.

    Truncation, when the prompt does not fit, removes tokens **from the left of the
    state**. It never touches the tail -- the criterion, the lettered options and
    "Answer:" -- because the answer slot is defined by the characters immediately
    before it, and trimming the tail would move the supervised position onto a token
    that is not an answer letter at all. The instruction preamble is kept at the front
    where it can be, so what is dropped is the oldest evidence, the way a human
    skimming a long record would drop it.

    Raises `TrainingError`, naming the decision, if the row is unlabelled, if the
    tokenizer has no stable answer slot for this prompt, or if the criterion alone is
    already longer than `max_length`.
    """
    if not decision.labelled:
        raise TrainingError(
            f"Decision {decision.id!r} has no answer_id, so there is no token to train "
            "on. Filter to labelled decisions before training, or label this row."
        )
    if max_length < 2:
        raise TrainingError(
            f"max_length must leave room for at least the answer token, got {max_length}."
        )

    criterion = decision.criterion
    prompt = full_prompt(decision.state, criterion)

    try:
        slots, _separator = answer_slots_for(tokenizer, decision)
        answer_token = slots[decision.answer_index]
    except SchemaError as error:  # answer_id not in this criterion; schema caught it
        raise TrainingError(f"Decision {decision.id!r}: {error}") from error

    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not prompt_ids:
        raise TrainingError(f"Decision {decision.id!r} encoded to zero prompt tokens.")

    budget = max_length - 1  # the answer token is not negotiable
    if len(prompt_ids) > budget:
        prompt_ids = _truncate_state_from_left(tokenizer, decision, prompt_ids, budget)

    input_ids = [*prompt_ids, answer_token]
    labels = [IGNORE_INDEX] * len(prompt_ids) + [answer_token]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def answer_slots_for(tokenizer: Tokenizer, decision: Decision) -> tuple[list[int], str]:
    """The answer-letter token ids at this decision's prompt boundary.

    A thin wrapper over `prompt.answer_slots` whose only job is to attach the decision
    id to the failure. A row whose answer slot cannot be resolved is a row whose loss
    would land on the wrong token, so it is reported, never quietly dropped.
    """
    prompt = full_prompt(decision.state, decision.criterion)
    try:
        return answer_slots(tokenizer, prompt, len(decision.criterion.options))
    except PromptError as error:
        raise TrainingError(
            f"Decision {decision.id!r} (criterion {decision.criterion.key!r}): {error}"
        ) from error


def _prefix_len(tokenizer: Tokenizer, text: str, prompt_ids: Sequence[int]) -> int:
    """How many leading tokens of `prompt_ids` `text` safely accounts for.

    The last token of `text` on its own is dropped before comparing: the character that
    follows can merge into it, so it is not reliably a token-level prefix. This is the
    same trim `prompt.state_prefix_ids` performs, for the same reason.
    """
    ids = list(tokenizer.encode(text, add_special_tokens=False))[:-1]
    if not ids or list(prompt_ids[: len(ids)]) != ids:
        return 0
    return len(ids)


def _truncate_state_from_left(
    tokenizer: Tokenizer,
    decision: Decision,
    prompt_ids: list[int],
    budget: int,
) -> list[int]:
    """Drop tokens from the left of the state until the prompt fits `budget`."""
    head_len = _prefix_len(tokenizer, head_text(decision.state), prompt_ids)
    if head_len == 0:
        # This tokenizer does not encode the head as a token-level prefix of the whole
        # prompt, so the state/criterion boundary cannot be located. Fall back to
        # keeping the rightmost tokens: the tail, and therefore the answer slot,
        # survives, which is the property that must not be lost.
        log.warning(
            "Decision %s: could not locate the state/criterion token boundary; "
            "truncating the whole prompt from the left instead.",
            decision.id,
        )
        return prompt_ids[-budget:]

    protected = len(prompt_ids) - head_len  # criterion, options, "Answer:" and the
    # final head token, which may have merged with the tail's first character.
    if protected > budget:
        tail = tail_text(decision.criterion)
        raise TrainingError(
            f"Decision {decision.id!r}: the criterion and its {len(decision.criterion.options)} "
            f"options need {protected} tokens, which is more than the {budget} available "
            f"under max_length. Raise max_length or shorten the option descriptions "
            f"(the tail is {len(tail)} characters). Truncating it would move the answer "
            "slot onto a token that is not an answer letter."
        )

    allowed_head = budget - protected
    preamble_len = _prefix_len(tokenizer, PREAMBLE, prompt_ids)
    if preamble_len >= allowed_head:
        # No room to keep the instruction and any evidence at all; keep the evidence
        # nearest the criterion, which is the part the criterion is most likely about.
        log.warning(
            "Decision %s: max_length leaves no room for the instruction preamble; "
            "keeping only the %d state tokens nearest the criterion.",
            decision.id, allowed_head,
        )
        kept_head = prompt_ids[head_len - allowed_head : head_len] if allowed_head else []
    else:
        keep_state = allowed_head - preamble_len
        kept_head = prompt_ids[:preamble_len] + prompt_ids[head_len - keep_state : head_len]

    truncated = kept_head + prompt_ids[head_len:]
    log.debug(
        "Decision %s: state truncated from the left, %d prompt tokens -> %d.",
        decision.id, len(prompt_ids), len(truncated),
    )
    return truncated


# ----------------------------------------------------------- option order


def reorder_options(decisions: Sequence[Decision], *, seed: int) -> list[Decision]:
    """Permute each decision's options, preserving `answer_id`.

    Defers to `typedecide.data.augment.randomise_option_order`, so the package has
    exactly one implementation of the permutation and one seeding scheme. (An inline
    fallback used to live here for the days before `data/` existed; it seeded by row
    position rather than by decision id, so had it ever run, the same seed would have
    produced a different order. It was unreachable and is gone.)
    """
    from ..data.augment import randomise_option_order

    return list(randomise_option_order(decisions, seed=seed))


# ------------------------------------------------------------------- dataset


class DecisionDataset:
    """An indexable view of decisions as masked, single-token-supervised examples.

    Duck-types `torch.utils.data.Dataset` (`__len__` and `__getitem__` are all
    `transformers.Trainer` asks for) without importing torch, which keeps the masking
    invariant testable with a stub tokenizer.

    **Option-order randomisation is applied per epoch.** `set_epoch` re-permutes every
    decision's options from `seed` and the epoch number, and the trainer calls it from
    an `on_epoch_begin` callback. Per epoch is affordable here because a permutation is
    a tuple rebuild, not a re-tokenisation, and it is strictly better than once up
    front: a model that sees each decision in a different order each epoch cannot learn
    "the answer is A". The manifest records which scheme ran.
    """

    def __init__(
        self,
        decisions: Sequence[Decision],
        tokenizer: Tokenizer,
        max_length: int,
        *,
        randomise_option_order: bool = False,
        seed: int = 0,
    ) -> None:
        if not decisions:
            raise TrainingError("No decisions to train on; the dataset is empty.")
        self._source: tuple[Decision, ...] = tuple(decisions)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._randomise = randomise_option_order
        self._seed = seed
        self._epoch = 0
        self._rows: tuple[Decision, ...] = self._source
        if self._randomise:
            self.set_epoch(0)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return build_example(self._tokenizer, self._rows[index], self._max_length)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """The decisions as currently ordered -- after this epoch's permutation."""
        return self._rows

    def set_epoch(self, epoch: int) -> None:
        """Re-permute the options for `epoch`. A no-op when randomisation is off."""
        self._epoch = epoch
        if not self._randomise:
            return
        # Mixing the epoch into the seed keeps every epoch different and the whole run
        # reproducible from the single `seed` in the config.
        self._rows = tuple(reorder_options(self._source, seed=self._seed * 1_000_003 + epoch))

    def preflight(self) -> dict[str, dict[str, int]]:
        """Resolve the answer slots once per distinct criterion, before training starts.

        Returns `{criterion key: {letter: token id}}` for the manifest, and raises
        `TrainingError` naming the offending decision if any criterion has no stable
        answer slot. Catching that now rather than at step 4,000 of a twelve-hour run
        is the entire point.
        """
        from ..schema import LETTERS

        table: dict[str, dict[str, int]] = {}
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for decision in self._source:
            criterion = decision.criterion
            signature = (criterion.question, criterion.option_ids)
            if signature in seen:
                continue
            seen.add(signature)
            slots, separator = answer_slots_for(self._tokenizer, decision)
            resolved = dict(zip(LETTERS, slots, strict=False))
            previous = table.get(criterion.key)
            if previous is not None and previous != resolved:
                log.warning(
                    "Criterion %s resolves to different answer-slot ids for different "
                    "option sets; the manifest records the last one seen.",
                    criterion.key,
                )
            table[criterion.key] = resolved
            log.debug(
                "Criterion %s: answer slots %s with separator %r.",
                criterion.key, resolved, separator,
            )
        return table


# ------------------------------------------------------------------ collator


def pad_batch(
    features: Sequence[dict[str, list[int]]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Right-pad a batch of examples to a common length.

    Right-padding is correct for *training*: the real tokens keep positions
    0..n-1, which is what the default `position_ids` give, and the padding is masked
    out of attention and carries `-100` labels so it contributes no loss. (Batched
    *scoring* is the opposite case and must left-pad with explicit position ids;
    that belongs to `evaluation/`, not here.)
    """
    if not features:
        raise TrainingError("Cannot collate an empty batch.")
    width = max(len(f["input_ids"]) for f in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        pad = width - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
        batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad)
    return batch


class AnswerTokenCollator:
    """Collates `build_example` outputs into the tensors `Trainer` expects.

    Written rather than borrowed from `transformers` so the padding rule above is
    stated in this repository, where the invariant it protects also lives.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise TrainingError(
                "Collating a batch needs torch, which is not installed. "
                "Install the training extra with `pip install typedecide[train]`."
            ) from error

        padded = pad_batch(features, self.pad_token_id)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
        }


def labelled_only(decisions: Iterable[Decision]) -> tuple[list[Decision], list[str]]:
    """Split decisions into the trainable ones and the ids of the unlabelled rest.

    The caller decides what to do about the unlabelled ones; this function never
    discards a row without handing back its id, because "n went in, fewer came out"
    with no names attached is how a bad dataset survives a run.
    """
    keep: list[Decision] = []
    dropped: list[str] = []
    for decision in decisions:
        if decision.labelled:
            keep.append(decision)
        else:
            dropped.append(decision.id)
    return keep, dropped


__all__ = [
    "IGNORE_INDEX",
    "AnswerTokenCollator",
    "DecisionDataset",
    "answer_slots_for",
    "build_example",
    "labelled_only",
    "pad_batch",
    "reorder_options",
]