"""The claim this module exists to make, tested against a fake model.

A model with a position prior answers by where an option sits rather than by what
the evidence says. The claim is that scoring every ordering and averaging the
probability *per option id* cancels that prior exactly, because under a cyclic or
full permutation set every option spends the same time in every slot.

So we inject a scorer that is deliberately broken in exactly that way -- a large
constant bonus for whatever is listed first, plus a small honest signal from the
evidence -- and check three things:

  * debias="none" gets it wrong, at exactly the rate a position-0 answerer would
  * debias="cyclic", "permutation" and "calibrated" all recover the right answer
  * order consistency and the letter histogram still report the prior, because
    averaging repaired the readout, not the model

No torch, no transformers, no downloads. The scorer is a `ScoreFn` like any other,
which is why `runner.evaluate_with_scorer` takes one.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # the package may not be pip-installed in CI
    sys.path.insert(0, str(_SRC))

from typedecide.errors import ConfigError, DataError  # noqa: E402
from typedecide.evaluation.readout import ScoreRequest  # noqa: E402
from typedecide.evaluation.runner import (  # noqa: E402
    NULL_STATE,
    evaluate_with_scorer,
    orderings,
)
from typedecide.schema import Criterion, Decision, Option  # noqa: E402

# ----------------------------------------------------------------- the fixture

CRITERION = Criterion(
    key="entailment",
    question="Assess the claim against the evidence.",
    options=(
        Option("supported", "The evidence establishes the claim"),
        Option("undetermined", "The evidence establishes neither"),
        Option("refuted", "The evidence establishes the opposite"),
    ),
)

# Gold rotates through all three option positions twice, so a model that always
# answers the first-listed option is right on exactly a third of the set: the same
# score an honest but useless model gets. That is the confusion this module exists
# to remove.
_GOLD_BY_STATE = {
    "The parcel left the depot and was signed for on arrival.": "supported",
    "The parcel left the depot; nothing is recorded after that.": "undetermined",
    "The parcel was returned to the depot undelivered.": "refuted",
    "The refund was issued and the customer confirmed receipt.": "supported",
    "The refund request is open and no decision is recorded.": "undetermined",
    "The refund was formally declined by the reviewer.": "refuted",
}

DECISIONS = [
    Decision(id=f"d{index}", state=state, criterion=CRITERION, answer_id=answer)
    for index, (state, answer) in enumerate(_GOLD_BY_STATE.items())
]


def _softmax(values: Sequence[float]) -> list[float]:
    top = max(values)
    exponentials = [math.exp(v - top) for v in values]
    total = sum(exponentials)
    return [v / total for v in exponentials]


class FakeScorer:
    """A model that mostly answers by position.

    Its logit for an option is `content_weight` if that option is the gold answer
    for the state it was given, plus `position_bonus` if the option happens to be
    listed first. With the bonus well above the weight, the first-listed option wins
    almost every single-ordering readout regardless of the evidence.

    Given the content-free state (`NULL_STATE`) no option is gold, so what comes back
    is the bare position prior -- exactly what contextual calibration divides out.
    """

    def __init__(self, *, position_bonus: float = 3.0, content_weight: float = 1.0) -> None:
        self.position_bonus = position_bonus
        self.content_weight = content_weight
        self.seen: list[ScoreRequest] = []
        self.batches = 0

    def __call__(self, requests: Sequence[ScoreRequest]) -> list[list[float]]:
        self.batches += 1
        out = []
        for request in requests:
            self.seen.append(request)
            gold = _GOLD_BY_STATE.get(request.state)
            logits = [
                (self.content_weight if option.id == gold else 0.0)
                + (self.position_bonus if position == 0 else 0.0)
                for position, option in enumerate(request.criterion.options)
            ]
            out.append(_softmax(logits))
        return out


# ------------------------------------------------------ the claim, stated as tests


def test_the_fake_model_is_only_broken_by_its_position_prior() -> None:
    """Control. With the bonus removed the same scorer is perfect under a single
    ordering, so everything that follows is caused by the prior and not by the
    fixture, the prompt, or the harness."""
    honest = FakeScorer(position_bonus=0.0)
    result = evaluate_with_scorer(DECISIONS, honest, debias="none")
    assert result.accuracy == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)


def test_single_ordering_readout_is_fooled_by_the_position_prior() -> None:
    """debias="none" scores each decision once, in the order the options are given,
    so the first-listed option wins every time. It is right only on the two rows
    whose gold happens to be listed first: 2/6.

    Note what the accuracy column looks like on its own -- 33%, which on a 3-option
    set is precisely chance and reads like a model that has learned nothing, rather
    than a model that has learned something very specific and very wrong."""
    scorer = FakeScorer()
    result = evaluate_with_scorer(DECISIONS, scorer, debias="none")

    assert result.accuracy == pytest.approx(2 / 6)
    assert result.balanced_accuracy == pytest.approx(1 / 3)
    # Every single pick landed on the first-listed option.
    assert result.letter_distribution == {"A": 1.0, "B": 0.0, "C": 0.0}
    # One ordering scored, so there is nothing for consistency to compare.
    assert result.order_consistency is None


@pytest.mark.parametrize("debias", ["cyclic", "permutation"])
def test_permutation_averaging_recovers_the_content_correct_answer(debias: str) -> None:
    """THE CLAIM. Same model, same data, same prompt -- only the averaging changes.

    Under a cyclic or full permutation set each option occupies each slot the same
    number of times, so the position bonus is added to every option's average
    equally and cancels out of the argmax. What is left is the evidence signal, and
    the evidence signal is correct.
    """
    scorer = FakeScorer()
    result = evaluate_with_scorer(DECISIONS, scorer, debias=debias)

    assert result.accuracy == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.n == len(DECISIONS)


def test_averaging_repairs_the_readout_but_the_model_is_still_broken() -> None:
    """The reason order consistency and the letter histogram are headline outputs.

    Cyclic averaging takes accuracy to 100%, and a report that stopped at accuracy
    would call this model fixed. It is not fixed: every individual forward pass still
    picks whatever is in slot A, so its answer flips with every rotation. Both of
    those facts are visible, and only because these two numbers are reported next to
    the accuracy that contradicts them.
    """
    scorer = FakeScorer()
    result = evaluate_with_scorer(DECISIONS, scorer, debias="cyclic")

    assert result.accuracy == pytest.approx(1.0)
    # Not one decision kept its answer when the options moved.
    assert result.order_consistency == pytest.approx(0.0)
    # And every one of the 18 forward passes chose the letter A.
    assert result.letter_distribution == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_calibration_fixes_each_individual_ordering_not_just_the_average() -> None:
    """Contextual calibration divides out the prior the model assigns with no
    evidence at all. The fake scorer's logits are content plus position, so dividing
    the softmax by the content-free softmax cancels the position term exactly -- and
    unlike plain averaging it makes each single ordering correct on its own.

    That shows up as order consistency 1.0 and a flat letter histogram: the picks are
    now spread across A, B and C because the answer follows the option, not the slot.
    """
    scorer = FakeScorer()
    result = evaluate_with_scorer(DECISIONS, scorer, debias="calibrated")

    assert result.accuracy == pytest.approx(1.0)
    assert result.order_consistency == pytest.approx(1.0)
    assert result.letter_distribution == {
        "A": pytest.approx(1 / 3),
        "B": pytest.approx(1 / 3),
        "C": pytest.approx(1 / 3),
    }


def test_the_content_free_prior_is_cached_per_question_and_ordering() -> None:
    """Six decisions share one question and one option set, so the three rotations
    give three distinct (question, ordering) pairs. Calibration must cost three extra
    forward passes, not eighteen."""
    scorer = FakeScorer()
    evaluate_with_scorer(DECISIONS, scorer, debias="calibrated")

    null_requests = [r for r in scorer.seen if r.state == NULL_STATE]
    assert len(null_requests) == 3
    # Three different arrangements of the same options.
    assert len({r.criterion.option_ids for r in null_requests}) == 3
    assert scorer.batches == 2  # one batch of real prompts, one of content-free ones


def test_averaging_is_per_option_id_not_per_position() -> None:
    """The averaged distribution is keyed by option id, so the winner survives being
    read back in the criterion's own order. If the accumulator were keyed by position
    the three rotations would just re-add the same slot bonuses and nothing would
    cancel; that failure mode looks exactly like debias="none"."""
    scorer = FakeScorer()
    cyclic = evaluate_with_scorer(DECISIONS, scorer, debias="cyclic")
    plain = evaluate_with_scorer(DECISIONS, FakeScorer(), debias="none")
    assert cyclic.accuracy > plain.accuracy
    assert cyclic.manifest["orderings_scored"] == 3
    assert cyclic.manifest["forward_passes"] == 18


def test_a_stronger_prior_still_cancels() -> None:
    """The prior does not have to be mild. At a bonus of 8 nats the first-listed
    option takes essentially all the probability mass in every single pass, and
    cyclic averaging still recovers the evidence signal."""
    scorer = FakeScorer(position_bonus=8.0)
    single = evaluate_with_scorer(DECISIONS, FakeScorer(position_bonus=8.0), debias="none")
    result = evaluate_with_scorer(DECISIONS, scorer, debias="cyclic")
    assert single.accuracy == pytest.approx(2 / 6)
    assert result.accuracy == pytest.approx(1.0)


def test_confidence_and_probability_outputs_are_well_formed() -> None:
    scorer = FakeScorer()
    result = evaluate_with_scorer(DECISIONS, scorer, debias="cyclic")
    assert result.ece is not None and 0.0 <= result.ece <= 1.0
    assert result.brier is not None and 0.0 <= result.brier <= 2.0
    assert set(result.per_criterion) == {"entailment"}
    assert result.per_criterion["entailment"] == pytest.approx(1.0)
    assert result.manifest["chance"] == pytest.approx(1 / 3)
    assert result.manifest["debias"] == "cyclic"
    assert len(result.manifest["fingerprint"]) == 64


def test_render_puts_the_diagnostic_numbers_in_the_report() -> None:
    result = evaluate_with_scorer(DECISIONS, FakeScorer(), debias="cyclic")
    text = result.render()
    assert "order consistency" in text
    assert "letters picked" in text
    assert "A:100%" in text


def test_render_says_so_when_one_ordering_proves_nothing() -> None:
    result = evaluate_with_scorer(DECISIONS, FakeScorer(), debias="none")
    assert "only one ordering was scored" in result.render()


# ------------------------------------------------------------- orderings and guards


def test_orderings_cover_every_slot_the_same_number_of_times() -> None:
    """The property that makes the cancellation exact. If any option appeared in a
    slot more often than another, the prior would not average out."""
    for debias, expected in (("cyclic", 4), ("permutation", 24)):
        orders = orderings(4, debias)
        assert len(orders) == expected
        assert len(set(orders)) == expected  # no ordering scored twice
        share = expected // 4
        for option in range(4):
            slots = [order.index(option) for order in orders]
            assert [slots.count(slot) for slot in range(4)] == [share] * 4


def test_orderings_none_is_the_identity() -> None:
    assert orderings(3, "none") == [(0, 1, 2)]


def test_orderings_refuses_an_unreasonable_permutation_set() -> None:
    with pytest.raises(ConfigError, match="cyclic"):
        orderings(9, "permutation")


def test_unknown_debias_mode_is_refused_by_name() -> None:
    with pytest.raises(ConfigError, match="rotations|not a mode"):
        evaluate_with_scorer(DECISIONS, FakeScorer(), debias="shuffle")


def test_unlabelled_decisions_are_refused() -> None:
    unlabelled = [Decision(id="u1", state="Some evidence.", criterion=CRITERION)]
    with pytest.raises(DataError, match="answer_id"):
        evaluate_with_scorer(unlabelled, FakeScorer())


def test_a_scorer_returning_the_wrong_shape_is_caught_not_averaged() -> None:
    """A silent shape mismatch would quietly score the wrong option, so the runner
    checks rather than trusts."""

    def truncating(requests: Sequence[ScoreRequest]) -> list[list[float]]:
        return [[0.5, 0.5] for _ in requests]  # two probabilities for three options

    with pytest.raises(ConfigError, match="3 options"):
        evaluate_with_scorer(DECISIONS, truncating, debias="none")

    def short(requests: Sequence[ScoreRequest]) -> list[list[float]]:
        return [[1 / 3, 1 / 3, 1 / 3] for _ in requests][:-1]

    with pytest.raises(ConfigError, match="one vector per request"):
        evaluate_with_scorer(DECISIONS, short, debias="none")
