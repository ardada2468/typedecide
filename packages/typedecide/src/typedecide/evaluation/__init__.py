"""Scoring a model the way a sceptic would.

`metrics` is pure arithmetic and imports nothing heavy. `readout` turns a model into
probabilities over a criterion's options. `runner` ties them together, scores each
decision under one or more option orderings, and reports balanced accuracy beside
the two numbers that actually expose a positional answerer: order consistency and
the distribution of letters picked.
"""

from __future__ import annotations

from .metrics import (
    accuracy,
    balanced_accuracy,
    brier_score,
    chance_accuracy,
    expected_calibration_error,
    letter_distribution,
    mean_group_balanced_accuracy,
    order_consistency,
)
from .readout import ScoreFn, ScoreRequest, TorchReadout, load_readout
from .runner import DEBIAS_MODES, NULL_STATE, EvalResult, evaluate, evaluate_with_scorer, orderings

__all__ = [
    "DEBIAS_MODES",
    "NULL_STATE",
    "EvalResult",
    "ScoreFn",
    "ScoreRequest",
    "TorchReadout",
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "chance_accuracy",
    "evaluate",
    "evaluate_with_scorer",
    "expected_calibration_error",
    "letter_distribution",
    "load_readout",
    "mean_group_balanced_accuracy",
    "order_consistency",
    "orderings",
]
