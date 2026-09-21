"""Quantisation tiers for the browser build.

Quantisation is the second-largest lever after pruning, and the only one that
can cost accuracy. This module keeps the tiers in one table so the size
arithmetic in `docs/SPEED.md` and the code that produces the files cannot drift
apart, and names each tier the way transformers.js does (`model_q4f16.onnx`), so
`web/models.json` can point at the output without renaming anything.

One thing is specific to this project and worth saying out loud: a typed readout
does not sample. It compares at most `len(LETTERS)` logits and takes an argmax.
Perplexity, which is what most quantisation write-ups report, is the wrong
number to watch. What matters is whether the *margin* between the top two answer
letters survives, and the honest way to find out is to score the same eval set
at each tier with `typedecide.evaluation` and compare `balanced_accuracy` and
`order_consistency`. A tier that keeps accuracy but collapses
`order_consistency` has turned the model into a position-guesser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExportError

log = logging.getLogger("typedecide.export.quantize")

DEFAULT_BLOCK_SIZE = 32


@dataclass(frozen=True, slots=True)
class QuantizationTier:
    """One tier: what it stores, what it computes in, and what it is likely to cost."""

    name: str
    suffix: str
    bits_per_weight: float
    compute_dtype: str
    summary: str

    @property
    def bytes_per_weight(self) -> float:
        return self.bits_per_weight / 8.0

    def bytes_for(self, parameters: int) -> float:
        """Weight bytes for a model of `parameters` parameters at this tier."""
        if parameters < 0:
            raise ExportError(f"A parameter count cannot be negative, got {parameters}.")
        return parameters * self.bytes_per_weight


# bits_per_weight for the 4-bit tiers is 4 + 16/block_size: four bits of weight
# plus one fp16 scale per block. At the default block size of 32 that is 4.5.
QUANTIZATION_MODES: dict[str, QuantizationTier] = {
    "fp32": QuantizationTier(
        name="fp32",
        suffix="",
        bits_per_weight=32.0,
        compute_dtype="fp32",
        summary=(
            "The export's own format. Lossless and unshippable: four bytes a weight is "
            "about four times what a browser should be asked to download."
        ),
    ),
    "fp16": QuantizationTier(
        name="fp16",
        suffix="fp16",
        bits_per_weight=16.0,
        compute_dtype="fp16",
        summary=(
            "Half the bytes, and the native compute dtype on WebGPU. Accuracy loss on a "
            "letter argmax is expected to be negligible; treat this as the reference "
            "point the smaller tiers are compared against, not as free."
        ),
    ),
    "q8": QuantizationTier(
        name="q8",
        suffix="q8",
        bits_per_weight=8.0,
        compute_dtype="fp32",
        summary=(
            "Per-channel int8 weights, float activations. The conservative choice when a "
            "4-bit tier moves accuracy more than you are willing to accept."
        ),
    ),
    "q4": QuantizationTier(
        name="q4",
        suffix="q4",
        bits_per_weight=4.0 + 16.0 / DEFAULT_BLOCK_SIZE,
        compute_dtype="fp32",
        summary=(
            "4-bit block-quantised MatMul weights with an fp16 scale per block of 32, "
            "float32 compute. Smaller download than q8, same compute cost."
        ),
    ),
    "q4f16": QuantizationTier(
        name="q4f16",
        suffix="q4f16",
        bits_per_weight=4.0 + 16.0 / DEFAULT_BLOCK_SIZE,
        compute_dtype="fp16",
        summary=(
            "4-bit weights and fp16 compute. What web/models.json asks for: the smallest "
            "download that keeps WebGPU on its fast path. Also the tier most likely to "
            "cost accuracy, so it is the one to measure rather than assume."
        ),
    ),
}


def quantize(
    model_path: Path | str,
    *,
    mode: str = "q4f16",
    output_path: Path | str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Path:
    """Quantise an exported ONNX model and return the file written.

    `model_path` is the `.onnx` file or the directory `export_onnx` wrote. The
    output is named the way transformers.js expects (`model_q4f16.onnx`) and
    placed beside the input unless `output_path` says otherwise.

    `mode="fp32"` is a no-op that returns the input unchanged, so a pipeline can
    pass the tier through without branching.

    Quantise **after** pruning, not before. Pruning removes rows; quantising
    first means spending bits on rows that are about to be deleted, and on a
    tied model it also forces the quantiser to pick one representation for a
    table that the input embedding and the unembedding no longer share.
    """
    tier = resolve_mode(mode)
    source = _resolve_model_file(Path(model_path))
    if tier.name == "fp32":
        log.info("mode 'fp32' is the export's own format; leaving %s alone", source)
        return source
    if block_size <= 0 or block_size % 16:
        raise ExportError(
            f"block_size must be a positive multiple of 16, got {block_size}. The 4-bit "
            "quantiser packs one fp16 scale per block and rejects other sizes."
        )

    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_{tier.suffix}.onnx")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == source:
        raise ExportError(
            f"Refusing to quantise {source} onto itself; pass a different `output_path`."
        )

    log.info("quantising %s -> %s (%s)", source, destination, tier.name)
    if tier.name == "q8":
        _quantize_int8(source, destination)
    elif tier.name in ("q4", "q4f16"):
        _quantize_4bit(source, destination, block_size=block_size, to_fp16=tier.name == "q4f16")
    elif tier.name == "fp16":
        _convert_fp16(source, destination)
    else:  # pragma: no cover - resolve_mode already rejected anything else
        raise ExportError(f"No implementation for tier {tier.name!r}.")
    return destination


def resolve_mode(mode: str) -> QuantizationTier:
    """The tier `mode` names, or an error listing the ones that exist."""
    tier = QUANTIZATION_MODES.get(mode)
    if tier is None:
        raise ExportError(
            f"Unknown quantisation mode {mode!r}. Available: "
            f"{', '.join(sorted(QUANTIZATION_MODES))}."
        )
    return tier


# ------------------------------------------------------------------ backends


def _quantize_4bit(source: Path, destination: Path, *, block_size: int, to_fp16: bool) -> None:
    onnx = _require("onnx")
    quantizer_module = _require("onnxruntime.quantization.matmul_4bits_quantizer")
    quantizer_class = getattr(quantizer_module, "MatMul4BitsQuantizer", None)
    if quantizer_class is None:
        raise ExportError(
            "onnxruntime.quantization has no MatMul4BitsQuantizer. 4-bit block "
            "quantisation needs onnxruntime >= 1.20: pip install 'typedecide[export]'"
        )
    proto = onnx.load(str(source))
    try:
        quantizer = quantizer_class(proto, block_size=block_size, is_symmetric=True)
        quantizer.process()
        result = quantizer.model.model
    except Exception as error:  # noqa: BLE001 - the quantiser raises assorted types
        raise ExportError(f"4-bit quantisation of {source} failed: {error}.") from error
    if to_fp16:
        result = _to_fp16(result)
    _save(onnx, result, destination)


def _quantize_int8(source: Path, destination: Path) -> None:
    quantization = _require("onnxruntime.quantization")
    try:
        quantization.quantize_dynamic(
            str(source),
            str(destination),
            weight_type=quantization.QuantType.QUInt8,
            op_types_to_quantize=["MatMul"],
            extra_options={"MatMulConstBOnly": True},
        )
    except Exception as error:  # noqa: BLE001
        raise ExportError(f"int8 quantisation of {source} failed: {error}.") from error


def _convert_fp16(source: Path, destination: Path) -> None:
    onnx = _require("onnx")
    _save(onnx, _to_fp16(onnx.load(str(source))), destination)


def _to_fp16(proto: Any) -> Any:
    """Cast float32 initialisers to float16, keeping the graph's inputs and
    outputs at their original types so the browser's I/O does not change."""
    for module_name in (
        "onnxconverter_common.float16",
        "onnxruntime.transformers.float16",
    ):
        try:
            import importlib

            converter = importlib.import_module(module_name)
        except ImportError:
            continue
        return converter.convert_float_to_float16(proto, keep_io_types=True)
    raise ExportError(
        "No float16 converter available. Install one with: "
        "pip install 'typedecide[export]' onnxconverter-common"
    )


# ----------------------------------------------------------------- odds and ends


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


def _require(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ExportError(
            f"{module!r} is needed to quantise but is not installed. Install it with: "
            "pip install 'typedecide[export]'"
        ) from error
