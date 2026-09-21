"""Metrics pinned to hand-computed values.

Every expected number in this file was worked out on paper and the working is in the
comment beside it. A metric test that computes the expected value with the same
expression as the implementation proves only that Python is deterministic.

Pure stdlib: no torch, no numpy, no model download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # the package may not be pip-installed in CI
    sys.path.insert(0, str(_SRC))

from typedecide.errors import DataError  # noqa: E402
from typedecide.evaluation.metrics import (  # noqa: E402
    accuracy,
    balanced_accuracy,
    brier_score,
    chance_accuracy,
    expected_calibration_error,
    letter_distribution,
    mean_group_balanced_accuracy,
    order_consistency,
)

# --------------------------------------------------------------- balanced accuracy


def test_balanced_accuracy_punishes_the_constant_answer() -> None:
    """The reason the headline number is balanced and not plain.

    Three 'a' and one 'b'; a model that always says 'a' is right 3 times in 4.
    Recall of a = 3/3 = 1, recall of b = 0/1 = 0, mean = 0.5.
    """
    gold = ["a", "a", "a", "b"]
    pred = ["a", "a", "a", "a"]
    assert accuracy(gold, pred) == pytest.approx(0.75)
    assert balanced_accuracy(gold, pred) == pytest.approx(0.5)


def test_balanced_accuracy_perfect_and_inverted() -> None:
    gold = ["a", "b", "c", "a"]
    assert balanced_accuracy(gold, list(gold)) == pytest.approx(1.0)
    assert balanced_accuracy(gold, ["b", "c", "a", "b"]) == pytest.approx(0.0)


def test_balanced_accuracy_ignores_classes_never_gold() -> None:
    # 'c' is predicted but never gold, so it has no recall to contribute.
    # recall a = 1/2, recall b = 1/1 -> 0.75
    gold = ["a", "a", "b"]
    pred = ["a", "c", "b"]
    assert balanced_accuracy(gold, pred) == pytest.approx(0.75)


def test_balanced_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(DataError, match="one to one"):
        balanced_accuracy(["a", "b"], ["a"])


def test_balanced_accuracy_rejects_empty() -> None:
    with pytest.raises(DataError, match="No labelled rows"):
        balanced_accuracy([], [])


# ------------------------------------------------------- mean group balanced acc.


def test_mean_group_balanced_accuracy_weights_groups_not_rows() -> None:
    """Group 'big' is perfect, group 'small' is at 0.5. The mean is 0.75 even though
    'big' carries four of the six rows -- which is the point of averaging groups."""
    gold = ["a", "a", "b", "b", "a", "b"]
    pred = ["a", "a", "b", "b", "a", "a"]
    groups = ["big", "big", "big", "big", "small", "small"]
    #   big:   recall a = 2/2, recall b = 2/2 -> 1.0
    #   small: recall a = 1/1, recall b = 0/1 -> 0.5
    assert mean_group_balanced_accuracy(gold, pred, groups) == pytest.approx(0.75)


def test_mean_group_balanced_accuracy_needs_a_key_per_row() -> None:
    with pytest.raises(DataError, match="one key per row"):
        mean_group_balanced_accuracy(["a", "b"], ["a", "b"], ["g"])


# ------------------------------------------------------------------------- ECE


def test_ece_ten_bins_hand_computed() -> None:
    """probs 0.9, 0.8, 0.3, 0.6 with outcomes hit, miss, miss, hit.

    With ten equal-width bins each value lands alone in its own bin:
      0.9 -> bin 9: acc 1, conf 0.9, gap 0.1
      0.8 -> bin 8: acc 0, conf 0.8, gap 0.8
      0.3 -> bin 3: acc 0, conf 0.3, gap 0.3
      0.6 -> bin 6: acc 1, conf 0.6, gap 0.4
    Each bin holds 1 of 4 rows, so ECE = (0.1 + 0.8 + 0.3 + 0.4) / 4 = 0.4
    """
    probs = [0.9, 0.8, 0.3, 0.6]
    correct = [True, False, False, True]
    assert expected_calibration_error(probs, correct, bins=10) == pytest.approx(0.4)


def test_ece_two_bins_hand_computed() -> None:
    """Same rows, two bins, so the bucketing actually does something.

      bin 1 [0.5, 1.0]: 0.9, 0.8, 0.6 -> acc 2/3 = 0.666..., conf 2.3/3 = 0.766...,
                        gap 0.1, weight 3/4  -> 0.075
      bin 0 [0.0, 0.5): 0.3          -> acc 0, conf 0.3, gap 0.3, weight 1/4 -> 0.075
      ECE = 0.15
    """
    probs = [0.9, 0.8, 0.3, 0.6]
    correct = [True, False, False, True]
    assert expected_calibration_error(probs, correct, bins=2) == pytest.approx(0.15)


def test_ece_is_zero_for_a_perfectly_calibrated_set() -> None:
    # Ten rows at confidence 0.7, seven of them right: acc 0.7, conf 0.7, gap 0.
    probs = [0.7] * 10
    correct = [True] * 7 + [False] * 3
    assert expected_calibration_error(probs, correct, bins=10) == pytest.approx(0.0)


def test_ece_is_one_for_confident_and_always_wrong() -> None:
    assert expected_calibration_error([1.0] * 5, [False] * 5, bins=10) == pytest.approx(1.0)


def test_ece_puts_one_point_zero_in_the_last_bin() -> None:
    # An off-by-one here would raise IndexError, so this is a boundary guard.
    assert expected_calibration_error([1.0, 0.0], [True, False], bins=10) == pytest.approx(0.0)


def test_ece_rejects_values_outside_the_unit_interval() -> None:
    with pytest.raises(DataError, match="outside"):
        expected_calibration_error([1.5], [True])


# ----------------------------------------------------------------------- Brier


def test_brier_hand_computed() -> None:
    """Two rows, three classes.

      [0.7, 0.2, 0.1] gold 0 -> (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
      [0.1, 0.6, 0.3] gold 1 -> 0.1^2 + (0.6-1)^2 + 0.3^2 = 0.01 + 0.16 + 0.09 = 0.26
      mean = 0.20
    """
    vectors = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]
    assert brier_score(vectors, [0, 1]) == pytest.approx(0.20)


def test_brier_bounds() -> None:
    assert brier_score([[1.0, 0.0, 0.0]], [0]) == pytest.approx(0.0)
    assert brier_score([[1.0, 0.0, 0.0]], [1]) == pytest.approx(2.0)


def test_brier_of_a_uniform_guess_is_one_minus_one_over_k() -> None:
    # k = 4: 3 * (0.25)^2 + (0.25 - 1)^2 = 0.1875 + 0.5625 = 0.75 = 1 - 1/4
    assert brier_score([[0.25] * 4], [0]) == pytest.approx(0.75)


def test_brier_accepts_rows_with_different_option_counts() -> None:
    # 0.0 for the certain 2-way row, 0.5 for the uniform 2-way row -> mean 0.25
    assert brier_score([[1.0, 0.0], [0.5, 0.5]], [0, 0]) == pytest.approx(0.25)


def test_brier_rejects_a_gold_index_outside_the_vector() -> None:
    with pytest.raises(DataError, match="gold index"):
        brier_score([[0.5, 0.5]], [2])


# ------------------------------------------------------------- order consistency


def test_order_consistency_counts_decisions_not_picks() -> None:
    picks = [
        ["a", "a", "a"],  # survived every ordering
        ["a", "b", "c"],  # answered by position
        ["b", "b", "b"],  # survived
        ["a", "a", "b"],  # flipped once, still counts as inconsistent
    ]
    assert order_consistency(picks) == pytest.approx(0.5)


def test_order_consistency_is_one_when_nothing_moves() -> None:
    assert order_consistency([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)


def test_order_consistency_rejects_a_decision_with_no_picks() -> None:
    with pytest.raises(DataError, match="no picks"):
        order_consistency([["a"], []])


# ----------------------------------------------------------- letter distribution


def test_letter_distribution_shows_the_letters_never_picked() -> None:
    """Forcing the key set is what makes the histogram a diagnosis: A:1.00 alone
    reads as a formatting quirk, A:1.00 B:0.00 C:0.00 reads as a position prior."""
    picked = ["A", "A", "A", "A"]
    assert letter_distribution(picked, letters="ABC") == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_letter_distribution_shares_sum_to_one() -> None:
    picked = ["A", "B", "B", "C"]
    distribution = letter_distribution(picked, letters="ABC")
    assert distribution == {"A": pytest.approx(0.25), "B": pytest.approx(0.5),
                            "C": pytest.approx(0.25)}
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_letter_distribution_infers_keys_when_not_given() -> None:
    assert letter_distribution(["B", "A"]) == {"A": 0.5, "B": 0.5}


# ------------------------------------------------------------------- chance rate


def test_chance_accuracy_mixes_option_counts() -> None:
    # (1/2 + 1/4) / 2 = 0.375
    assert chance_accuracy([2, 4]) == pytest.approx(0.375)
    assert chance_accuracy([3, 3, 3]) == pytest.approx(1 / 3)


def test_chance_accuracy_rejects_impossible_counts() -> None:
    with pytest.raises(DataError, match="cannot have"):
        chance_accuracy([0])


# ------------------------------------------------------------ refusals, by name


def test_every_metric_refuses_empty_or_misaligned_input() -> None:
    from typedecide.evaluation import metrics as m

    cases: list[tuple[object, str]] = [
        (lambda: m.accuracy(["a"], []), "1 gold labels and 0 predictions"),
        (lambda: m.balanced_accuracy([], []), "No labelled rows to score"),
        (lambda: m.mean_group_balanced_accuracy(["a"], ["a"], []), "0 group keys for 1 rows"),
        (lambda: m.expected_calibration_error([0.5], []), "1 confidences and 0 outcomes"),
        (lambda: m.expected_calibration_error([], []), "No rows to calibrate over"),
        (lambda: m.expected_calibration_error([0.5], [True], bins=0), "bins must be at least 1"),
        (lambda: m.expected_calibration_error([1.5], [True]), r"outside \[0, 1\]"),
        (lambda: m.brier_score([[1.0]], []), "1 probability vectors and 0 gold indices"),
        (lambda: m.brier_score([], []), "No rows to score"),
        (lambda: m.brier_score([[]], [0]), "Row 0 has an empty probability vector"),
        (lambda: m.brier_score([[0.5, 0.5]], [2]), "gold index 2 but its vector has 2"),
        (lambda: m.order_consistency([]), "No decisions to check"),
        (lambda: m.order_consistency([["a"], []]), "position 1 has no picks"),
        (lambda: m.letter_distribution([]), "No picks to build"),
        (lambda: m.chance_accuracy([]), "No rows to compute a chance rate"),
        (lambda: m.chance_accuracy([2, 0]), "cannot have 0 options"),
    ]
    for call, message in cases:
        with pytest.raises(DataError, match=message):
            call()  # type: ignore[operator]


def test_letter_distribution_counts_a_letter_outside_the_forced_key_set() -> None:
    from typedecide.evaluation.metrics import letter_distribution

    assert letter_distribution(["A", "C"], letters=["A", "B"]) == {"A": 0.5, "B": 0.0, "C": 0.5}
    assert letter_distribution(["B", "B"]) == {"B": 1.0}
