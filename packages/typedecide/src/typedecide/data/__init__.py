"""Loading, validating, splitting and permuting decisions.

The bring-your-own-data surface. Nothing here imports torch, so `typedecide.data`
stays usable in a CI job with no GPU and no model cache.
"""

from __future__ import annotations

from .augment import all_permutations, all_rotations, randomise_option_order
from .loaders import FieldMapping, load_decisions, write_decisions
from .split import group_split
from .validate import (
    DEFAULT_MAX_STATE_TOKENS,
    DataReport,
    Finding,
    estimate_tokens,
    validate,
)

__all__ = [
    "DEFAULT_MAX_STATE_TOKENS",
    "DataReport",
    "FieldMapping",
    "Finding",
    "all_permutations",
    "all_rotations",
    "estimate_tokens",
    "group_split",
    "load_decisions",
    "randomise_option_order",
    "validate",
    "write_decisions",
]
