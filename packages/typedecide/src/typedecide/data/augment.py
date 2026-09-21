"""Permuting the options, because position is the easiest signal in the dataset.

If every row lists the correct answer first, the cheapest hypothesis consistent with
the training data is "answer A", and a model will find it long before it finds the
task. Randomising the order at training time removes that shortcut; enumerating the
orders at evaluation time measures whether any of it survived.

`answer_id` is carried through every permutation untouched -- the answer is an option
id, not a position, which is the whole reason the schema works this way.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Sequence
from dataclasses import replace

from ..errors import DataError
from ..schema import Decision

LOGGER = logging.getLogger("typedecide.data.augment")

__all__ = ["all_permutations", "all_rotations", "randomise_option_order"]

#: 8! is 40320 variants of one row. Past this, enumerate rotations instead.
MAX_PERMUTATION_OPTIONS = 8


def randomise_option_order(decisions: Sequence[Decision], *, seed: int) -> list[Decision]:
    """One new ordering per decision. Same length, same ids, same answers.

    The per-row generator is seeded from `(seed, decision.id)` rather than from the
    row's position, so a row gets the same ordering whichever dataset it is part of
    and whatever order it was loaded in.
    """
    out: list[Decision] = []
    for decision in decisions:
        count = len(decision.criterion.options)
        rng = random.Random(f"{seed}:{decision.id}")
        order = list(range(count))
        rng.shuffle(order)
        out.append(_reordered(decision, order, suffix=None))
    LOGGER.debug("randomised option order for %d decisions (seed=%d)", len(out), seed)
    return out


def all_rotations(decision: Decision) -> list[Decision]:
    """The n cyclic shifts of one decision, identity first.

    A cyclic set puts every option in every position exactly once, so a position prior
    contributes the same total to each option and cancels when the probabilities are
    averaged per option id. n forward passes instead of n!.
    """
    count = len(decision.criterion.options)
    return [
        _reordered(decision, [(i + shift) % count for i in range(count)], suffix=f"r{shift}")
        for shift in range(count)
    ]


def all_permutations(decision: Decision) -> list[Decision]:
    """Every ordering of one decision's options, identity first.

    Exact rather than cyclic, and factorially expensive: use it to audit a small
    sample, and `all_rotations` for anything you run over a whole eval set.
    """
    count = len(decision.criterion.options)
    if count > MAX_PERMUTATION_OPTIONS:
        raise DataError(
            f"Decision {decision.id!r} has {count} options, so all_permutations would build "
            f"{count}! variants of a single row. Fix: use all_rotations(), which puts every "
            f"option in every position with {count} variants, or sample orderings instead."
        )
    return [
        _reordered(decision, list(order), suffix=f"p{index}")
        for index, order in enumerate(itertools.permutations(range(count)))
    ]


def _reordered(decision: Decision, order: Sequence[int], *, suffix: str | None) -> Decision:
    criterion = decision.criterion.reordered(order)
    meta = dict(decision.meta)
    meta["meta_option_order"] = ",".join(criterion.option_ids)
    if suffix is None:
        return replace(decision, criterion=criterion, meta=meta)
    meta["meta_source_id"] = decision.id
    return replace(decision, id=f"{decision.id}#{suffix}", criterion=criterion, meta=meta)
