"""The arithmetic of scoring, and nothing else.

Everything here is a pure function over plain Python sequences. No model, no
tokenizer, no torch, no file system. That separation is deliberate: the numbers a
reader will quote from a run are computed by code that a unit test can pin to
hand-worked values without downloading half a gigabyte of weights first.

Two of these functions matter more than accuracy does. `order_consistency` asks
whether the model gives the same answer when the options are listed in a different
order, and a letter histogram (see `letter_distribution`) asks which letters it
picked. A model that has learned "answer A" scores exactly chance on a balanced set
and is invisible in an accuracy column, but it lights up both of these.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import DataError

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "chance_accuracy",
    "expected_calibration_error",
    "letter_distribution",
    "mean_group_balanced_accuracy",
    "order_consistency",
]


def _check_pairwise(gold: Sequence[str], pred: Sequence[str]) -> None:
    if len(gold) != len(pred):
        raise DataError(
            f"Got {len(gold)} gold labels and {len(pred)} predictions. "
            "They must line up one to one, in the same order."
        )
    if not gold:
        raise DataError("No labelled rows to score. Check the eval split is not empty.")


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Plain hit rate. Reported for continuity with other tools, not for trust:
    on an unbalanced set a constant answer can look respectable."""
    _check_pairwise(gold, pred)
    hits = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    return hits / len(gold)


def balanced_accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Mean per-class recall over the classes that actually occur in `gold`.

    Each gold class contributes equally regardless of how many rows carry it, so a
    model that always answers the majority class scores 1/k rather than the class's
    prevalence. Classes that never appear as gold are not scored -- there is no
    recall to measure for them.
    """
    _check_pairwise(gold, pred)
    recalls = []
    for label in sorted(set(gold)):
        members = [i for i, g in enumerate(gold) if g == label]
        recalls.append(sum(1 for i in members if pred[i] == label) / len(members))
    return sum(recalls) / len(recalls)


def mean_group_balanced_accuracy(
    gold: Sequence[str], pred: Sequence[str], groups: Sequence[str]
) -> float:
    """Balanced accuracy computed inside each group, then averaged over groups.

    Groups are usually criterion keys. Averaging over groups rather than rows stops
    the largest criterion from deciding the headline number on its own.
    """
    _check_pairwise(gold, pred)
    if len(groups) != len(gold):
        raise DataError(
            f"Got {len(groups)} group keys for {len(gold)} rows; one key per row is needed."
        )
    per_group = []
    for group in sorted(set(groups)):
        members = [i for i, key in enumerate(groups) if key == group]
        per_group.append(
            balanced_accuracy([gold[i] for i in members], [pred[i] for i in members])
        )
    return sum(per_group) / len(per_group)


def expected_calibration_error(
    probs: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """Equal-width binned ECE.

    Confidences are dropped into `bins` equal-width buckets of [0, 1]; a bucket
    covers [lo, hi) except the last, which includes 1.0. For each non-empty bucket
    we take |mean accuracy - mean confidence|, and average those over buckets
    weighted by how many rows landed in each. Zero means the model's stated
    confidence matches how often it is right.
    """
    if len(probs) != len(correct):
        raise DataError(
            f"Got {len(probs)} confidences and {len(correct)} outcomes; they must line up."
        )
    if not probs:
        raise DataError("No rows to calibrate over.")
    if bins < 1:
        raise DataError(f"bins must be at least 1, got {bins}.")
    for value in probs:
        if not 0.0 <= value <= 1.0:
            raise DataError(f"Confidence {value!r} is outside [0, 1]; ECE needs probabilities.")

    totals = [0] * bins
    hits = [0] * bins
    confidence = [0.0] * bins
    for value, is_right in zip(probs, correct, strict=True):
        index = min(int(value * bins), bins - 1)
        totals[index] += 1
        hits[index] += 1 if is_right else 0
        confidence[index] += value

    n = len(probs)
    error = 0.0
    for index in range(bins):
        if totals[index] == 0:
            continue
        gap = abs(hits[index] / totals[index] - confidence[index] / totals[index])
        error += (totals[index] / n) * gap
    return error


def brier_score(prob_vectors: Sequence[Sequence[float]], gold_indices: Sequence[int]) -> float:
    """Multi-class Brier score: mean over rows of the squared error of the whole
    probability vector against a one-hot gold.

    For one row with vector p and gold class g this is sum_k (p_k - [k == g])^2,
    which runs from 0 (certain and right) to 2 (certain and wrong). It is not halved,
    so a k-way uniform guess scores 1 - 1/k. Vectors may differ in length across
    rows, because criteria may differ in how many options they offer.
    """
    if len(prob_vectors) != len(gold_indices):
        raise DataError(
            f"Got {len(prob_vectors)} probability vectors and {len(gold_indices)} gold indices."
        )
    if not prob_vectors:
        raise DataError("No rows to score.")

    total = 0.0
    for row, (vector, gold) in enumerate(zip(prob_vectors, gold_indices, strict=True)):
        if not vector:
            raise DataError(f"Row {row} has an empty probability vector.")
        if not 0 <= gold < len(vector):
            raise DataError(
                f"Row {row} names gold index {gold} but its vector has {len(vector)} entries."
            )
        total += sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(vector))
    return total / len(prob_vectors)


def order_consistency(picks_per_decision: Sequence[Sequence[str]]) -> float:
    """Share of decisions that got the same answer under every ordering scored.

    This is the number that catches a positional answerer. Accuracy averaged over
    orderings can look fine while the individual answers flip every time the options
    move; 1.0 here means the model's answer is a property of the evidence, and a low
    value means it is a property of the layout.

    A decision scored under a single ordering is trivially consistent and counts as
    such, so the figure is only meaningful when more than one ordering was scored --
    `runner.evaluate` reports None rather than 1.0 in that case.
    """
    if not picks_per_decision:
        raise DataError("No decisions to check for order consistency.")
    consistent = 0
    for index, picks in enumerate(picks_per_decision):
        if not picks:
            raise DataError(f"Decision at position {index} has no picks to compare.")
        if len(set(picks)) == 1:
            consistent += 1
    return consistent / len(picks_per_decision)


def letter_distribution(
    picked_letters: Sequence[str], *, letters: Sequence[str] | None = None
) -> dict[str, float]:
    """Share of picks that landed on each answer letter.

    Pass `letters` to force the key set so letters that were never picked appear as
    0.0 -- a histogram reading A:1.00 B:0.00 C:0.00 is the whole diagnosis, and it
    only reads that way if B and C are present.
    """
    if not picked_letters:
        raise DataError("No picks to build a letter distribution from.")
    keys = list(letters) if letters is not None else sorted(set(picked_letters))
    counts = dict.fromkeys(keys, 0)
    for letter in picked_letters:
        if letter not in counts:
            counts[letter] = 0
        counts[letter] += 1
    n = len(picked_letters)
    return {key: counts[key] / n for key in counts}


def chance_accuracy(option_counts: Sequence[int]) -> float:
    """What a uniform guesser scores on this set: the mean of 1/n over the rows.

    Quoted beside accuracy because 62% means one thing against 50% chance and
    another against 33%.
    """
    if not option_counts:
        raise DataError("No rows to compute a chance rate for.")
    for count in option_counts:
        if count < 1:
            raise DataError(f"A decision cannot have {count} options.")
    return sum(1.0 / count for count in option_counts) / len(option_counts)
