"""The domain types, plus regressions for the conversion and fingerprint bugs."""

from __future__ import annotations

import json
from typing import Any

import pytest

from typedecide.errors import SchemaError
from typedecide.schema import (
    LETTERS,
    MAX_OPTIONS,
    Criterion,
    Decision,
    Option,
    criterion_from_dict,
    decision_from_dict,
    decision_to_dict,
    fingerprint,
    stable_id,
)

OPTIONS = [{"id": "no", "description": "No"}, {"id": "yes", "description": "Yes"}]
ROW: dict[str, Any] = {"id": "d1", "state": "evidence", "question": "Is it?", "options": OPTIONS}


def _criterion() -> Criterion:
    return Criterion("k", "Is it?", (Option("no", "No"), Option("yes", "Yes")))


def test_option_and_criterion_validation() -> None:
    with pytest.raises(SchemaError, match="lowercase alphanumeric"):
        Option("Not OK", "x")
    with pytest.raises(SchemaError, match="no description"):
        Option("ok", "   ")
    with pytest.raises(SchemaError, match="Criterion key"):
        Criterion("Bad Key", "q", (Option("a", "A"), Option("b", "B")))
    with pytest.raises(SchemaError, match="no question text"):
        Criterion("k", " ", (Option("a", "A"), Option("b", "B")))
    with pytest.raises(SchemaError, match="between 2 and"):
        Criterion("k", "q", (Option("a", "A"),))
    with pytest.raises(SchemaError, match="twice"):
        Criterion("k", "q", (Option("a", "A"), Option("a", "B")))
    assert len(LETTERS) == MAX_OPTIONS


def test_reordered_and_index_of() -> None:
    criterion = _criterion()
    assert criterion.reordered([1, 0]).option_ids == ("yes", "no")
    with pytest.raises(SchemaError, match="permutation"):
        criterion.reordered([0, 0])
    with pytest.raises(SchemaError, match="is not an option"):
        criterion.index_of("maybe")


def test_decision_validation_and_properties() -> None:
    with pytest.raises(SchemaError, match="non-empty id"):
        Decision(id=" ", state="s", criterion=_criterion())
    with pytest.raises(SchemaError, match="empty state"):
        Decision(id="d", state="", criterion=_criterion())
    with pytest.raises(SchemaError, match="is not an option"):
        Decision(id="d", state="s", criterion=_criterion(), answer_id="maybe")
    unlabelled = Decision(id="d", state="s", criterion=_criterion())
    assert not unlabelled.labelled and unlabelled.grouping == "d"
    with pytest.raises(SchemaError, match="no answer to index"):
        _ = unlabelled.answer_index
    labelled = Decision(id="d", state="s", criterion=_criterion(), answer_id="yes", group_id="g")
    assert labelled.answer_index == 1 and labelled.grouping == "g"


def test_from_dict_accepts_each_spelling_of_the_gold() -> None:
    assert decision_from_dict({**ROW, "answer_id": "yes"}).answer_id == "yes"
    assert decision_from_dict({**ROW, "answer": "no"}).answer_id == "no"
    assert decision_from_dict({**ROW, "answer_index": 1}).answer_id == "yes"
    assert decision_from_dict({**ROW, "label": "0"}).answer_id == "no"
    assert decision_from_dict(ROW).answer_id is None
    assert decision_from_dict({**ROW, "label": None}).answer_id is None


@pytest.mark.parametrize("field", ["answer_index", "label"])
@pytest.mark.parametrize("bad", ["second", 1.5, True, [1], float("inf")])
def test_unparseable_position_is_a_schema_error(field: str, bad: Any) -> None:
    """int('second') raised a bare ValueError, which the contract forbids and which the
    CLI does not catch, so a bad cell became a traceback."""
    with pytest.raises(SchemaError, match=field):
        decision_from_dict({**ROW, field: bad})


def test_out_of_range_position_is_a_schema_error() -> None:
    with pytest.raises(SchemaError, match="outside the 2 options"):
        decision_from_dict({**ROW, "answer_index": 2})
    with pytest.raises(SchemaError, match="outside the options"):
        decision_from_dict({**ROW, "label": -1})


def test_from_dict_reports_missing_fields_and_bad_options() -> None:
    with pytest.raises(SchemaError, match=r"missing \['options', 'state'\]"):
        decision_from_dict({"question": "q"})
    with pytest.raises(SchemaError, match="needs an 'id' and a 'description'"):
        decision_from_dict({**ROW, "options": [{"id": "a"}, {"id": "b"}]})
    with pytest.raises(SchemaError, match="needs an 'id' and a 'description'"):
        criterion_from_dict({"question": "q", "options": [object(), object()]})


def test_missing_id_is_content_addressed() -> None:
    row = {k: v for k, v in ROW.items() if k != "id"}
    first, second = decision_from_dict(row), decision_from_dict(dict(row))
    assert first.id == second.id == stable_id(row) and len(first.id) == 20


def test_meta_cannot_overwrite_a_schema_field() -> None:
    """`**decision.meta` was spread last, so meta={'answer_id': ...} changed the gold of
    every dataset written from that decision."""
    decision = Decision(id="d", state="s", criterion=_criterion(), answer_id="yes",
                        meta={"answer_id": "no", "id": "evil", "meta_team": "billing"})
    payload = decision_to_dict(decision)
    assert payload["answer_id"] == "yes" and payload["id"] == "d"
    assert payload["meta_team"] == "billing"
    assert decision_from_dict(payload) == decision


def test_round_trip_preserves_everything() -> None:
    decision = decision_from_dict({**ROW, "answer_id": "yes", "group_id": 7, "meta_x": [1, 2]})
    assert decision.group_id == "7"
    assert decision_from_dict(json.loads(json.dumps(decision_to_dict(decision)))) == decision


def test_fingerprint_ignores_order_even_with_duplicate_ids() -> None:
    """Sorting on id alone is stable, so two rows sharing an id hashed in load order and
    the same dataset could fingerprint two ways."""
    a = Decision(id="dup", state="first", criterion=_criterion(), answer_id="yes")
    b = Decision(id="dup", state="second", criterion=_criterion(), answer_id="no")
    c = Decision(id="other", state="third", criterion=_criterion())
    assert fingerprint([a, b, c]) == fingerprint([c, b, a]) == fingerprint(iter([b, a, c]))
    assert fingerprint([a, c]) != fingerprint([b, c])


def test_fingerprint_is_pinned_for_unique_ids() -> None:
    """The tie-break must not move the hash of any valid dataset: manifests written by
    0.1.0 have to keep matching. This is the 0.1.0 algorithm, restated."""
    import hashlib

    rows = [
        Decision(id=f"d{n}", state=f"s{n}", criterion=_criterion(), answer_id="yes",
                 meta={"meta_n": n})
        for n in (3, 1, 2)
    ]
    digest = hashlib.sha256()
    for decision in sorted(rows, key=lambda d: d.id):
        digest.update(
            json.dumps(decision_to_dict(decision), ensure_ascii=False, sort_keys=True).encode()
        )
    assert fingerprint(rows) == digest.hexdigest()


def test_fingerprint_tolerates_unserialisable_meta() -> None:
    import datetime

    decision = Decision(id="d", state="s", criterion=_criterion(),
                        meta={"meta_seen": datetime.date(2026, 1, 2)})
    assert len(fingerprint([decision])) == 64
