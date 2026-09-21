"""Augmentation tests. The invariant under test is: the answer survives the shuffle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typedecide.data.augment import (  # noqa: E402
    all_permutations,
    all_rotations,
    randomise_option_order,
)
from typedecide.errors import DataError  # noqa: E402
from typedecide.schema import Criterion, Decision, Option  # noqa: E402


def criterion(n: int) -> Criterion:
    return Criterion(
        key="urgency",
        question="How urgent is this ticket?",
        options=tuple(Option(f"o{i}", f"Option number {i}") for i in range(n)),
    )


def make(ident: str = "t1", n: int = 4, answer: str | None = "o2") -> Decision:
    return Decision(
        id=ident,
        state=f"Evidence for {ident}, one sentence long.",
        criterion=criterion(n),
        answer_id=answer,
        group_id="g1",
    )


# ------------------------------------------------------------- randomise_option_order


def test_randomising_preserves_ids_answer_and_identity() -> None:
    rows = [make(f"t{i}") for i in range(20)]
    out = randomise_option_order(rows, seed=0)
    assert len(out) == len(rows)
    for before, after in zip(rows, out, strict=True):
        assert after.id == before.id
        assert after.answer_id == before.answer_id
        assert set(after.criterion.option_ids) == set(before.criterion.option_ids)
        assert after.state == before.state
        assert after.grouping == before.grouping
        # the answer still points at the same option text, wherever it now sits
        assert (
            after.criterion.options[after.answer_index].description
            == before.criterion.options[before.answer_index].description
        )


def test_randomising_actually_reorders_something() -> None:
    rows = [make(f"t{i}", n=6) for i in range(20)]
    out = randomise_option_order(rows, seed=0)
    moved = [
        a
        for a, b in zip(out, rows, strict=True)
        if a.criterion.option_ids != b.criterion.option_ids
    ]
    assert len(moved) > len(rows) // 2


def test_randomising_is_deterministic_and_seed_sensitive() -> None:
    rows = [make(f"t{i}", n=5) for i in range(20)]
    first = [d.criterion.option_ids for d in randomise_option_order(rows, seed=1)]
    again = [d.criterion.option_ids for d in randomise_option_order(rows, seed=1)]
    other = [d.criterion.option_ids for d in randomise_option_order(rows, seed=2)]
    assert first == again
    assert first != other


def test_a_rows_order_does_not_depend_on_its_position_in_the_list() -> None:
    target = make("t9", n=5)
    alone = randomise_option_order([target], seed=4)[0]
    in_a_crowd = randomise_option_order(
        [make(f"t{i}", n=5) for i in range(9)] + [target], seed=4
    )[-1]
    assert alone.criterion.option_ids == in_a_crowd.criterion.option_ids


def test_randomising_records_the_order_in_meta() -> None:
    out = randomise_option_order([make()], seed=0)[0]
    assert out.meta["meta_option_order"] == ",".join(out.criterion.option_ids)


def test_unlabelled_rows_are_shuffled_too() -> None:
    out = randomise_option_order([make(answer=None)], seed=0)[0]
    assert out.answer_id is None
    assert len(out.criterion.options) == 4


def test_randomising_an_empty_list_is_an_empty_list() -> None:
    assert randomise_option_order([], seed=0) == []


# --------------------------------------------------------------------------- rotations


def test_rotations_put_every_option_in_every_position_once() -> None:
    base = make(n=4)
    rotations = all_rotations(base)
    assert len(rotations) == 4
    for position in range(4):
        seen = {r.criterion.options[position].id for r in rotations}
        assert seen == set(base.criterion.option_ids)


def test_rotations_start_from_the_original_order_and_keep_the_answer() -> None:
    base = make(n=3)
    rotations = all_rotations(base)
    assert rotations[0].criterion.option_ids == base.criterion.option_ids
    assert all(r.answer_id == base.answer_id for r in rotations)
    assert len({r.id for r in rotations}) == len(rotations)
    assert all(r.meta["meta_source_id"] == base.id for r in rotations)
    assert all(r.grouping == base.grouping for r in rotations)


def test_rotation_answer_index_moves_but_the_option_does_not() -> None:
    base = make(n=4, answer="o0")
    indices = {r.answer_index for r in all_rotations(base)}
    assert indices == {0, 1, 2, 3}


# ------------------------------------------------------------------------ permutations


def test_permutations_enumerate_every_order() -> None:
    base = make(n=4)
    permutations = all_permutations(base)
    assert len(permutations) == 24
    assert len({p.criterion.option_ids for p in permutations}) == 24
    assert permutations[0].criterion.option_ids == base.criterion.option_ids
    assert all(p.answer_id == base.answer_id for p in permutations)
    assert len({p.id for p in permutations}) == 24


def test_permutations_refuse_a_factorial_explosion() -> None:
    base = make(n=9)
    with pytest.raises(DataError) as caught:
        all_permutations(base)
    message = str(caught.value)
    assert "all_rotations" in message
    assert "9 options" in message


def test_rotations_are_affordable_where_permutations_are_not() -> None:
    assert len(all_rotations(make(n=9))) == 9
