"""Split tests: determinism, no leakage, and a refusal rather than a silent hole."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typedecide.data.split import group_split  # noqa: E402
from typedecide.errors import DataError  # noqa: E402
from typedecide.schema import Criterion, Decision, Option  # noqa: E402

URGENCY = Criterion(
    key="urgency",
    question="How urgent is this ticket?",
    options=(Option("low", "Can wait a week"), Option("high", "Needs attention today")),
)


def make(ident: str, answer: str, group: str) -> Decision:
    return Decision(
        id=ident,
        state=f"Evidence for {ident}, which is its own sentence.",
        criterion=URGENCY,
        answer_id=answer,
        group_id=group,
    )


def both_classes_everywhere(n_groups: int = 10) -> list[Decision]:
    """Every ticket is asked once and answered both ways, so any split covers both."""
    rows: list[Decision] = []
    for i in range(n_groups):
        rows.append(make(f"t{i}-low", "low", f"g{i}"))
        rows.append(make(f"t{i}-high", "high", f"g{i}"))
    return rows


def ids(decisions: list[Decision]) -> set[str]:
    return {d.id for d in decisions}


# --------------------------------------------------------------------- happy path


def test_split_is_deterministic_for_a_seed() -> None:
    rows = both_classes_everywhere()
    first = group_split(rows, seed=7)
    second = group_split(rows, seed=7)
    assert ids(first[0]) == ids(second[0])
    assert ids(first[1]) == ids(second[1])


def test_split_does_not_depend_on_the_input_order() -> None:
    rows = both_classes_everywhere()
    forward = group_split(rows, seed=3)
    backward = group_split(list(reversed(rows)), seed=3)
    assert ids(forward[1]) == ids(backward[1])


def test_no_group_lands_on_both_sides() -> None:
    train, evaluation = group_split(both_classes_everywhere(), seed=1)
    assert {d.grouping for d in train}.isdisjoint({d.grouping for d in evaluation})
    assert ids(train) | ids(evaluation) == ids(both_classes_everywhere())
    assert not ids(train) & ids(evaluation)


def test_eval_side_is_roughly_the_requested_fraction() -> None:
    rows = both_classes_everywhere(50)
    _, evaluation = group_split(rows, eval_fraction=0.2, seed=0)
    assert 0.15 <= len(evaluation) / len(rows) <= 0.30


def test_rows_keep_their_input_order_within_a_side() -> None:
    rows = both_classes_everywhere()
    train, evaluation = group_split(rows, seed=5)
    order = [d.id for d in rows]
    assert [d.id for d in train] == [i for i in order if i in ids(train)]
    assert [d.id for d in evaluation] == [i for i in order if i in ids(evaluation)]


def test_different_seeds_can_give_different_splits() -> None:
    rows = both_classes_everywhere(20)
    splits = {frozenset(ids(group_split(rows, seed=s)[1])) for s in range(6)}
    assert len(splits) > 1


def test_unlabelled_rows_are_split_without_complaint() -> None:
    rows = [
        *both_classes_everywhere(6),
        Decision(id="u1", state="An unlabelled ticket.", criterion=URGENCY, group_id="gu"),
    ]
    train, evaluation = group_split(rows, seed=0)
    assert len(train) + len(evaluation) == len(rows)


# ------------------------------------------------------------------------ refusals


def _one_class_per_group() -> list[Decision]:
    """'high' lives only in g0, so any one-group eval side misses a class."""
    rows = [make("t0", "high", "g0")]
    rows += [make(f"t{i}", "low", f"g{i}") for i in range(1, 5)]
    return rows


def test_refuses_when_a_class_would_be_absent_from_eval() -> None:
    with pytest.raises(DataError) as caught:
        group_split(_one_class_per_group(), eval_fraction=0.01, seed=0)
    message = str(caught.value)
    assert "zero examples in eval" in message
    assert "urgency/" in message
    assert "eval_fraction" in message and "seed" in message


def test_a_class_confined_to_one_group_says_so() -> None:
    with pytest.raises(DataError) as caught:
        group_split(_one_class_per_group(), eval_fraction=0.01, seed=0)
    assert "one group, so no group split can ever put it on both sides" in str(caught.value)


def test_refuses_an_empty_dataset() -> None:
    with pytest.raises(DataError, match="empty dataset"):
        group_split([])


def test_refuses_when_everything_shares_one_group() -> None:
    rows = [make(f"t{i}", "low" if i % 2 else "high", "all") for i in range(6)]
    with pytest.raises(DataError) as caught:
        group_split(rows)
    assert "every row on one side" in str(caught.value)
    assert "group_id" in str(caught.value)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_refuses_an_impossible_eval_fraction(fraction: float) -> None:
    with pytest.raises(DataError, match="strictly between 0 and 1"):
        group_split(both_classes_everywhere(), eval_fraction=fraction)


def test_never_hands_back_an_empty_train_side() -> None:
    rows = both_classes_everywhere(3)
    train, evaluation = group_split(rows, eval_fraction=0.99, seed=0)
    assert train and evaluation


def test_unlabelled_rows_split_without_a_class_check() -> None:
    from typedecide.data.split import group_split as split
    from typedecide.schema import Criterion, Decision, Option

    criterion = Criterion("k", "q?", (Option("a", "A"), Option("b", "B")))
    rows = [Decision(id=f"u{i}", state=f"s{i}", criterion=criterion) for i in range(10)]
    train, held = split(rows, eval_fraction=0.3, seed=0)
    assert len(train) == 7 and len(held) == 3

    mixed = [*rows, Decision(id="l", state="sl", criterion=criterion, answer_id="a", group_id="u0")]
    seeds = [s for s in range(20) if _holds(split, mixed, s)]
    assert seeds, "a labelled row that lands in eval satisfies the class check"


def _holds(split: object, rows: list[object], seed: int) -> bool:
    from typedecide.errors import DataError as Refused

    try:
        _, held = split(rows, eval_fraction=0.3, seed=seed)  # type: ignore[operator]
    except Refused:
        return False
    return any(d.id == "l" for d in held)
