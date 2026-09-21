"""Merge, prune, and export a decoder ONNX Runtime Web can load.

The browser wants the graph shape transformers.js expects: a merged decoder with
past-key inputs, exported at an opset ORT Web actually implements. We produce
that with Optimum, having first merged any LoRA adapter and -- the part that
matters for speed -- sliced the unembedding down to the answer letters.

Everything heavy (`torch`, `transformers`, `peft`, `optimum`, `onnx`) is
imported inside the function that needs it, so `import typedecide` stays fast.

Read `docs/SPEED.md` before tuning anything here. In particular: a stock export
returns logits for **every** position, which on a 512-token prefill is a
311 MB fp32 tensor copied out of the GPU on one call.
`trim_logits_to_last_position` is the fix, and pruning is what shrinks what is
left.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._compat import from_pretrained_with_dtype
from ..errors import ExportError
from ..schema import LETTERS, MAX_OPTIONS
from .prune import (
    SlotMap,
    answer_letter_slots,
    build_slot_map,
    prune_lm_head,
    validate_keep,
    write_slot_map,
)

log = logging.getLogger("typedecide.export.onnx")

ONNX_TASK = "text-generation-with-past"
MANIFEST_FILENAME = "export_manifest.json"
STAGING_DIRNAME = "_merged"
MIN_OPSET = 14


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """One ONNX export.

    `prune_head_to` is the list of **original vocabulary ids** to keep, in answer
    order -- normally `answer_letter_token_ids(base_model, n_options)`. Leave it
    `None` to export the full unembedding. When it is set, the export also writes
    `slot_map.json` next to the model, and without that file the resulting
    weights are uninterpretable: the readout must index rows `0..K-1`, not
    vocabulary ids.

    `keep_merged` keeps the intermediate merged checkpoint on disk. It is large
    and only useful for debugging a suspicious export.
    """

    base_model: str
    adapter: Path | None
    output_dir: Path
    opset: int = 17
    prune_head_to: Sequence[int] | None = None
    keep_merged: bool = False

    def __post_init__(self) -> None:
        if not self.base_model or not self.base_model.strip():
            raise ExportError("An export needs a base model id or path; `base_model` is empty.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.adapter is not None:
            object.__setattr__(self, "adapter", Path(self.adapter))
        if self.opset < MIN_OPSET:
            raise ExportError(
                f"opset {self.opset} is below {MIN_OPSET}; ONNX Runtime Web needs at least "
                f"{MIN_OPSET} for a decoder with a past-key cache. 17 is what we ship."
            )
        if self.prune_head_to is not None:
            keep = validate_keep(self.prune_head_to)
            object.__setattr__(self, "prune_head_to", tuple(keep))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "adapter": None if self.adapter is None else str(self.adapter),
            "output_dir": str(self.output_dir),
            "opset": self.opset,
            "prune_head_to": None if self.prune_head_to is None else list(self.prune_head_to),
            "keep_merged": self.keep_merged,
        }


def export_onnx(config: ExportConfig) -> Path:
    """Export the model and return the directory holding it.

    Order matters. The adapter is merged first, because pruning a model that
    still has an adapter attached would prune the base weights and leave the
    adapter pointing at rows that no longer exist. Pruning happens before the
    ONNX trace, because the trace freezes the output width into the graph.
    """
    torch = _require("torch")
    transformers = _require("transformers")

    started = _now()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading %s", config.base_model)
    tokenizer = _load_tokenizer(transformers, config.base_model)
    model = _load_model(transformers, torch, config.base_model)

    if config.adapter is not None:
        if not config.adapter.exists():
            raise ExportError(
                f"No adapter at {config.adapter}. Point `adapter` at the directory "
                "`finetune` wrote (it holds adapter_config.json), or pass None to export "
                "the base model."
            )
        peft = _require("peft")
        log.info("merging adapter %s", config.adapter)
        try:
            model = peft.PeftModel.from_pretrained(model, str(config.adapter)).merge_and_unload()
        except Exception as error:  # noqa: BLE001 - peft raises a wide variety here
            raise ExportError(
                f"Merging {config.adapter} into {config.base_model} failed: {error}. The "
                "adapter and the base model must be the same architecture; check "
                "manifest.json beside the adapter for the base model it was trained on."
            ) from error

    slot_map: SlotMap | None = None
    if config.prune_head_to is not None:
        slot_map = _prune(model, tokenizer, config)
        write_slot_map(slot_map, output_dir)

    staging = output_dir / STAGING_DIRNAME
    log.info("staging the merged checkpoint at %s", staging)
    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)

    try:
        _run_optimum_export(staging, output_dir, config.opset)
        manifest = {
            "config": config.to_manifest(),
            "task": ONNX_TASK,
            "slot_map": None if slot_map is None else slot_map.to_dict(),
            "pruned": slot_map is not None,
            "logits_are_row_indices": slot_map is not None,
            "can_generate_text": slot_map is None,
            "versions": _versions(),
            "started": started,
            "finished": _now(),
        }
        (output_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if not config.keep_merged:
            shutil.rmtree(staging, ignore_errors=True)

    log.info("exported to %s", output_dir)
    return output_dir


def _prune(model: Any, tokenizer: Any, config: ExportConfig) -> SlotMap:
    """Slice the head and describe the slice, or explain why we cannot."""
    keep = list(config.prune_head_to or ())
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    keep = validate_keep(keep, vocab_size=vocab_size)

    separator = _probe_separator(tokenizer, keep)
    prune_lm_head(model, keep)
    return build_slot_map(
        keep,
        model_id=config.base_model,
        separator=separator,
        letters=LETTERS[: len(keep)] if len(keep) <= len(LETTERS) else None,
        original_vocab_size=vocab_size,
    )


def _probe_separator(tokenizer: Any, keep: Sequence[int]) -> str | None:
    """Re-derive the separator the kept ids came from, so the browser can send
    the same prompt boundary. `None` means we could not reproduce them, which is
    worth recording honestly rather than guessing."""
    if not 2 <= len(keep) <= MAX_OPTIONS:
        return None
    try:
        probed, separator = answer_letter_slots(tokenizer, len(keep))
    except ExportError as error:
        log.warning("could not probe the answer-letter boundary: %s", error)
        return None
    if list(probed) != list(keep):
        log.warning(
            "the ids being kept (%s) are not the ones this tokenizer produces at the prompt "
            "boundary (%s). Recording separator as null; the readout and the export may be "
            "scoring different tokens.",
            list(keep),
            list(probed),
        )
        return None
    return separator


def trim_logits_to_last_position(
    model_path: Path | str, *, output_path: Path | str | None = None
) -> Path:
    """Make the graph return only the final position's logits.

    A stock Optimum export returns `[batch, seq, vocab]` for the whole sequence.
    A typed readout reads exactly one position, so on a prefill every other
    position is computed, serialised, and discarded. At 512 tokens and fp32 that
    is 512 x 151936 x 4 = 311,168,512 bytes copied out of the GPU on a single
    call. Appending a `Slice` on the sequence axis turns that into one position.

    Decode steps already have `seq == 1`, so this changes nothing there; it is a
    prefill fix. Downstream code that slices the last position itself keeps
    working, because slicing the last of one position is a no-op.

    Returns the path written. Needs `onnx`.
    """
    onnx = _require("onnx")
    numpy = _require("numpy")
    from onnx import helper, numpy_helper

    source = _resolve_model_file(Path(model_path))
    destination = Path(output_path) if output_path is not None else source
    graph = (proto := onnx.load(str(source))).graph

    target = next((o for o in graph.output if o.name == "logits"), None)
    if target is None:
        if not graph.output:
            raise ExportError(f"{source} has no graph outputs to trim.")
        target = graph.output[0]
        log.warning("no output named 'logits'; trimming %r instead", target.name)
    name = target.name
    inner = f"{name}_all_positions"

    dims = target.type.tensor_type.shape.dim
    if len(dims) != 3:
        raise ExportError(
            f"Expected {name!r} to be a 3-D [batch, sequence, vocab] tensor; it has "
            f"{len(dims)} dimensions. This graph is not shaped the way a decoder export is, "
            "so trimming the sequence axis would cut the wrong thing."
        )

    full = helper.make_tensor_value_info(
        inner, target.type.tensor_type.elem_type, [None, None, None]
    )
    for node in graph.node:
        for i, produced in enumerate(node.output):
            if produced == name:
                node.output[i] = inner
        for i, consumed in enumerate(node.input):
            if consumed == name:
                node.input[i] = inner

    constants = [
        numpy_helper.from_array(numpy.array([-1], dtype=numpy.int64), "typedecide_trim_starts"),
        numpy_helper.from_array(
            numpy.array([numpy.iinfo(numpy.int64).max], dtype=numpy.int64), "typedecide_trim_ends"
        ),
        numpy_helper.from_array(numpy.array([1], dtype=numpy.int64), "typedecide_trim_axes"),
    ]
    graph.initializer.extend(constants)
    graph.value_info.append(full)
    graph.node.append(
        helper.make_node(
            "Slice",
            [inner, *(c.name for c in constants)],
            [name],
            name="typedecide_trim_last_position",
        )
    )
    dims[1].ClearField("dim_param")
    dims[1].dim_value = 1

    try:
        onnx.checker.check_model(proto, full_check=False)
    except Exception as error:  # noqa: BLE001 - onnx raises its own checker errors
        raise ExportError(
            f"The trimmed graph does not pass onnx.checker: {error}. Nothing was written; "
            f"{source} is untouched."
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(onnx, proto, destination)
    log.info("trimmed %s to the last position -> %s", source, destination)
    return destination


# ------------------------------------------------------------------ plumbing


def _load_tokenizer(transformers: Any, model_id: str) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as error:  # noqa: BLE001
        raise ExportError(
            f"Could not load the tokenizer for {model_id!r}: {error}."
        ) from error


def _load_model(transformers: Any, torch: Any, model_id: str) -> Any:
    """Load in float32. Quantisation is a later, explicit step (`quantize`), not
    something the loader should decide."""
    # The retry for transformers < 4.56 (which spells it torch_dtype) lives inside the
    # helper, and therefore inside this `try`. It used to sit in an `except TypeError:`
    # arm, where a failure of the retry itself escaped as a raw traceback: an exception
    # raised inside one `except` arm is not seen by its siblings.
    try:
        return from_pretrained_with_dtype(
            transformers.AutoModelForCausalLM.from_pretrained, model_id, torch.float32
        )
    except Exception as error:  # noqa: BLE001 - hub, network and config failures alike
        raise ExportError(f"Could not load {model_id!r}: {error}.") from error


def _run_optimum_export(source: Path, output_dir: Path, opset: int) -> None:
    exporters = _require("optimum.exporters.onnx")
    main_export = getattr(exporters, "main_export", None)
    if main_export is None:
        raise ExportError(
            "optimum.exporters.onnx has no `main_export`. Install a supported version: "
            "pip install 'typedecide[export]'"
        )
    log.info("exporting ONNX (task=%s, opset=%d) to %s", ONNX_TASK, opset, output_dir)
    try:
        main_export(
            model_name_or_path=str(source),
            output=str(output_dir),
            task=ONNX_TASK,
            opset=opset,
        )
    except Exception as error:  # noqa: BLE001 - optimum raises many unrelated types
        raise ExportError(
            f"The ONNX export failed: {error}. If the model has a pruned head, check that "
            "config.vocab_size still describes the *input* embedding -- pruning must not "
            "change it, and a mismatched vocab_size makes the exporter build out-of-range "
            "dummy inputs."
        ) from error


def _resolve_model_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [path / "onnx" / "model.onnx", path / "model.onnx"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ExportError(
        f"No ONNX model at {path}. Looked for {', '.join(str(c) for c in candidates)}. "
        "Pass the .onnx file directly, or the directory `export_onnx` wrote."
    )


def _save(onnx: Any, proto: Any, destination: Path) -> None:
    """Write, falling back to external data for graphs past the 2 GB protobuf cap."""
    if proto.ByteSize() > 2_000_000_000:
        onnx.save_model(
            proto,
            str(destination),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{destination.name}.data",
        )
        return
    onnx.save_model(proto, str(destination))


def _versions() -> dict[str, str]:
    import importlib.metadata as metadata

    found: dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "optimum", "onnx", "onnxruntime"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def _now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _require(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ExportError(
            f"{module!r} is needed to export ONNX but is not installed. Install the "
            "exporter's dependencies with: pip install 'typedecide[export]' "
            "'typedecide[train]'"
        ) from error
