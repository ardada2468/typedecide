"""The types every other module speaks.

A `Decision` is one question asked about one state, with a closed set of options and
optionally the answer. That is the whole domain model. Everything else in this
library loads these, validates them, trains on them, scores them, or exports a model
that answers them.

The types are frozen because a decision that changes between loading and scoring is
a bug that is very hard to see in a metrics table.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import SchemaError

# One answer letter per option. Twenty is the ceiling because that is how many
# letters we render; beyond it a tokenizer is unlikely to give one clean token each.
LETTERS = "ABCDEFGHIJKLMNOPQRST"
MIN_OPTIONS = 2
MAX_OPTIONS = len(LETTERS)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Option:
    """One permitted answer. `id` is what a caller receives; `description` is what
    the model reads."""

    id: str
    description: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.match(self.id):
            raise SchemaError(
                f"Option id {self.id!r} must be lowercase alphanumeric with _ . or -, "
                "start with a letter or digit, and be at most 64 characters."
            )
        if not self.description.strip():
            raise SchemaError(f"Option {self.id!r} has no description for the model to read.")


@dataclass(frozen=True, slots=True)
class Criterion:
    """One question and the closed set of answers it permits."""

    key: str
    question: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.match(self.key):
            raise SchemaError(
                f"Criterion key {self.key!r} must be lowercase alphanumeric with _ . or -."
            )
        if not self.question.strip():
            raise SchemaError(f"Criterion {self.key!r} has no question text.")
        if not MIN_OPTIONS <= len(self.options) <= MAX_OPTIONS:
            raise SchemaError(
                f"Criterion {self.key!r} has {len(self.options)} options; "
                f"between {MIN_OPTIONS} and {MAX_OPTIONS} are required."
            )
        ids = [o.id for o in self.options]
        duplicate = next((i for i in ids if ids.count(i) > 1), None)
        if duplicate is not None:
            raise SchemaError(f"Criterion {self.key!r} lists option id {duplicate!r} twice.")

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self.options)

    def index_of(self, option_id: str) -> int:
        try:
            return self.option_ids.index(option_id)
        except ValueError:
            raise SchemaError(
                f"{option_id!r} is not an option of criterion {self.key!r}; "
                f"it permits {', '.join(self.option_ids)}."
            ) from None

    def reordered(self, order: Sequence[int]) -> Criterion:
        """The same criterion with its options permuted. Used to cancel position priors."""
        if sorted(order) != list(range(len(self.options))):
            raise SchemaError("A reordering must be a permutation of the option indices.")
        return Criterion(self.key, self.question, tuple(self.options[i] for i in order))


@dataclass(frozen=True, slots=True)
class Decision:
    """One labelled (or unlabelled) decision.

    `group_id` marks decisions that must not be split across train and eval -- most
    often every criterion asked about one state. Leaving it unset means the decision
    forms its own group, which is the safe default.
    """

    id: str
    state: str
    criterion: Criterion
    answer_id: str | None = None
    group_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise SchemaError("Every decision needs a non-empty id.")
        if not self.state.strip():
            raise SchemaError(f"Decision {self.id!r} has an empty state.")
        if self.answer_id is not None:
            self.criterion.index_of(self.answer_id)  # raises if not a permitted option

    @property
    def labelled(self) -> bool:
        return self.answer_id is not None

    @property
    def answer_index(self) -> int:
        if self.answer_id is None:
            raise SchemaError(f"Decision {self.id!r} has no answer to index.")
        return self.criterion.index_of(self.answer_id)

    @property
    def grouping(self) -> str:
        """The key a group-aware split uses. Falls back to the decision's own id."""
        return self.group_id or self.id


# --------------------------------------------------------------------- conversion


def criterion_from_dict(payload: dict[str, Any]) -> Criterion:
    try:
        options = tuple(
            Option(id=str(o["id"]), description=str(o["description"]))
            for o in payload["options"]
        )
    except (KeyError, TypeError) as error:
        raise SchemaError(
            "Each option needs an 'id' and a 'description'. "
            f"Got: {json.dumps(payload.get('options'), ensure_ascii=False, default=str)[:160]}"
        ) from error
    return Criterion(
        key=str(payload.get("key") or payload.get("criterion") or "decision"),
        question=str(payload["question"]),
        options=options,
    )


def decision_from_dict(payload: dict[str, Any]) -> Decision:
    missing = {"state", "question", "options"} - payload.keys()
    if missing:
        raise SchemaError(f"Row is missing {sorted(missing)}.")
    criterion = criterion_from_dict(payload)

    answer = payload.get("answer_id", payload.get("answer"))
    if answer is None and payload.get("answer_index") is not None:
        index = _positional_index(payload["answer_index"], "answer_index")
        if not 0 <= index < len(criterion.options):
            raise SchemaError(
                f"answer_index {index} is outside the {len(criterion.options)} options given."
            )
        answer = criterion.options[index].id
    if answer is None and payload.get("label") is not None:
        # openjev-style fixtures carry the gold as a positional label.
        index = _positional_index(payload["label"], "label")
        if not 0 <= index < len(criterion.options):
            raise SchemaError(f"label {index} is outside the options given.")
        answer = criterion.options[index].id

    return Decision(
        id=str(payload.get("id") or stable_id(payload)),
        state=str(payload["state"]),
        criterion=criterion,
        answer_id=None if answer is None else str(answer),
        group_id=None if payload.get("group_id") is None else str(payload["group_id"]),
        meta={k: v for k, v in payload.items() if k.startswith("meta_")},
    )


def _positional_index(raw: Any, field_name: str) -> int:
    """A 0-based option position, or a `SchemaError` -- never a bare `ValueError`."""
    if isinstance(raw, bool):
        raise SchemaError(f"{field_name} {raw!r} is a boolean, not a 0-based option position.")
    try:
        index = int(raw)
        exact = float(raw) == index
    except (TypeError, ValueError, OverflowError):
        raise SchemaError(
            f"{field_name} {raw!r} is not a whole number. Fix: give the 0-based position of "
            "the gold option, or use 'answer_id' with the option's id."
        ) from None
    if not exact:
        raise SchemaError(f"{field_name} {raw!r} is not a whole number.")
    return index


def decision_to_dict(decision: Decision) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": decision.id,
        "state": decision.state,
        "key": decision.criterion.key,
        "question": decision.criterion.question,
        "options": [{"id": o.id, "description": o.description} for o in decision.criterion.options],
        "answer_id": decision.answer_id,
        "group_id": decision.group_id,
    }
    # Metadata rides along but never overwrites a schema field: a `meta` entry named
    # "answer_id" must not be able to change the gold of a written dataset.
    for key, value in decision.meta.items():
        if key not in payload:
            payload[key] = value
    return payload


def stable_id(payload: dict[str, Any]) -> str:
    """A content-addressed id, so re-running an unlabelled import is idempotent."""
    body = json.dumps(
        {k: payload.get(k) for k in ("state", "question", "options")},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


def fingerprint(decisions: Iterable[Decision]) -> str:
    """One hash over a dataset, recorded in run manifests so results name their input."""
    digest = hashlib.sha256()
    # Sorted on (id, body), not id alone: with duplicate ids a stable sort would keep
    # load order and the same rows could hash two ways. With unique ids -- every valid
    # dataset -- this is byte-identical to sorting on id.
    bodies = sorted(
        (
            decision.id,
            json.dumps(
                decision_to_dict(decision), ensure_ascii=False, sort_keys=True, default=str
            ),
        )
        for decision in decisions
    )
    for _, body in bodies:
        digest.update(body.encode("utf-8"))
    return digest.hexdigest()
