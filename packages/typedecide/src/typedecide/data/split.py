"""Splitting on groups, not on rows.

Every criterion asked about one ticket shares that ticket's evidence. Split those
across train and eval and the model has read the eval set during training, which
shows up as a score nobody can reproduce in production. So the unit of the split is
`Decision.grouping`, and a split that would hide a whole class from eval is refused
rather than quietly returned.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence

from ..errors import DataError
from ..schema import Decision

LOGGER = logging.getLogger("typedecide.data.split")

__all__ = ["group_split"]


def group_split(
    decisions: Sequence[Decision],
    *,
    eval_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Decision], list[Decision]]:
    """Split into (train, eval) on `Decision.grouping`.

    Deterministic given `seed`: groups are sorted by key before being shuffled, so the
    result does not depend on the order the decisions were loaded in. Rows keep their
    input order within each side.

    Raises `DataError` if the eval side would hold zero examples of some (criterion,
    answer) class, because a class absent from eval is a class the score says nothing
    about.
    """
    if not decisions:
        raise DataError(
            "Cannot split an empty dataset. Fix: check the loader's source and any filter "
            "applied after loading."
        )
    if not 0.0 < eval_fraction < 1.0:
        raise DataError(
            f"eval_fraction={eval_fraction} must be strictly between 0 and 1. "
            "Fix: pass something like 0.2 for a 20% eval side."
        )

    by_group: dict[str, list[Decision]] = defaultdict(list)
    for decision in decisions:
        by_group[decision.grouping].append(decision)

    keys = sorted(by_group)
    if len(keys) < 2:
        raise DataError(
            f"All {len(decisions)} rows share the group {keys[0]!r}, so a group split would put "
            "every row on one side. Fix: set group_id to the thing that must not be split -- "
            "usually the state or the source ticket -- or leave it unset so each row is its own "
            "group."
        )

    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)

    total = len(decisions)
    target = eval_fraction * total
    eval_keys: list[str] = []
    taken = 0
    for key in shuffled:
        if taken >= target or len(eval_keys) == len(shuffled) - 1:
            break
        eval_keys.append(key)
        taken += len(by_group[key])
    if not eval_keys:
        eval_keys = [shuffled[0]]

    held = set(eval_keys)
    _refuse_if_class_missing(decisions, by_group, held, eval_fraction, seed)

    train = [d for d in decisions if d.grouping not in held]
    evaluation = [d for d in decisions if d.grouping in held]
    LOGGER.info(
        "split %d rows in %d groups into %d train / %d eval (seed=%d)",
        total,
        len(keys),
        len(train),
        len(evaluation),
        seed,
    )
    return train, evaluation


def _refuse_if_class_missing(
    decisions: Sequence[Decision],
    by_group: dict[str, list[Decision]],
    held: set[str],
    eval_fraction: float,
    seed: int,
) -> None:
    """Every (criterion key, answer id) present in the data must survive into eval."""
    present: dict[tuple[str, str], int] = defaultdict(int)
    groups_with: dict[tuple[str, str], set[str]] = defaultdict(set)
    for decision in decisions:
        if decision.answer_id is None:
            continue
        key = (decision.criterion.key, decision.answer_id)
        present[key] += 1
        groups_with[key].add(decision.grouping)

    if not present:
        return

    in_eval: set[tuple[str, str]] = set()
    for group in held:
        for decision in by_group[group]:
            if decision.answer_id is not None:
                in_eval.add((decision.criterion.key, decision.answer_id))

    missing = sorted(set(present) - in_eval)
    if not missing:
        return

    lines = []
    for criterion_key, answer_id in missing:
        n = present[(criterion_key, answer_id)]
        groups = len(groups_with[(criterion_key, answer_id)])
        detail = f"  {criterion_key}/{answer_id}: {n} row(s) across {groups} group(s)"
        if groups == 1:
            detail += " -- one group, so no group split can ever put it on both sides"
        lines.append(detail)

    raise DataError(
        f"A split at eval_fraction={eval_fraction} with seed={seed} would leave "
        f"{len(missing)} class(es) with zero examples in eval, so the score would say nothing "
        "about them:\n"
        + "\n".join(lines)
        + "\nFix: raise eval_fraction, try another seed, collect more examples of those "
        "classes, or -- if a class genuinely lives in one group -- merge or drop it before "
        "splitting."
    )
