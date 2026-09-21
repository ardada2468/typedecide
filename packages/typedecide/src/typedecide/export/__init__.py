"""Shipping a typed-readout model to a browser.

Three steps, in this order:

1. `export_onnx` merges the adapter, prunes the unembedding to the answer
   letters, and traces the decoder to ONNX with a past-key cache.
2. `trim_logits_to_last_position` makes the graph return one position instead of
   the whole sequence.
3. `quantize` writes the tier `web/models.json` asks for.

A pruned model emits row indices, not vocabulary ids, so `slot_map.json` travels
with it. It also cannot generate text -- see `prune_lm_head` for exactly what is
lost and why. `docs/SPEED.md` holds the arithmetic, and marks which claims are
measured and which are derived.

Nothing here imports torch, transformers, optimum, onnx or onnxruntime at module
level; each function asks for what it needs and says which extra to install.
"""

from __future__ import annotations

from .onnx import ExportConfig, export_onnx, trim_logits_to_last_position
from .prune import (
    SLOT_MAP_FILENAME,
    SlotMap,
    answer_letter_slots,
    answer_letter_token_ids,
    build_slot_map,
    lm_head_savings,
    prune_lm_head,
    read_slot_map,
    select_rows,
    validate_keep,
    write_slot_map,
)
from .quantize import QUANTIZATION_MODES, QuantizationTier, quantize, resolve_mode

__all__ = [
    "QUANTIZATION_MODES",
    "SLOT_MAP_FILENAME",
    "ExportConfig",
    "QuantizationTier",
    "SlotMap",
    "answer_letter_slots",
    "answer_letter_token_ids",
    "build_slot_map",
    "export_onnx",
    "lm_head_savings",
    "prune_lm_head",
    "quantize",
    "read_slot_map",
    "resolve_mode",
    "select_rows",
    "trim_logits_to_last_position",
    "validate_keep",
    "write_slot_map",
]
