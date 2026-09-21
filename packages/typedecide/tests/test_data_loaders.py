"""Loader tests, weighted towards the malformed CSV a ticketing system produces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typedecide.data.loaders import (  # noqa: E402
    FieldMapping,
    load_decisions,
    write_decisions,
)
from typedecide.errors import DataError  # noqa: E402

ROW = {
    "id": "tkt-1",
    "state": "Customer cannot log in after the password reset email expired.",
    "question": "How urgent is this ticket?",
    "options": [
        {"id": "low", "description": "Can wait a week"},
        {"id": "high", "description": "Needs attention today"},
    ],
    "answer_id": "high",
    "group_id": "tkt-1",
}


def _jsonl(tmp_path: Path, *rows: dict[str, object], name: str = "tickets.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _csv(tmp_path: Path, text: str, name: str = "tickets.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------- jsonl/json


def test_jsonl_round_trip(tmp_path: Path) -> None:
    decisions = load_decisions(_jsonl(tmp_path, ROW))
    assert len(decisions) == 1
    only = decisions[0]
    assert only.id == "tkt-1"
    assert only.answer_id == "high"
    assert only.criterion.option_ids == ("low", "high")
    assert only.grouping == "tkt-1"

    out = tmp_path / "out" / "copy.jsonl"
    write_decisions(decisions, out)
    assert load_decisions(out)[0] == only


def test_write_decisions_json_array(tmp_path: Path) -> None:
    decisions = load_decisions(_jsonl(tmp_path, ROW))
    out = tmp_path / "copy.json"
    write_decisions(decisions, out)
    assert isinstance(json.loads(out.read_text()), list)
    assert load_decisions(out) == decisions


def test_jsonl_bad_line_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(ROW) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(DataError) as caught:
        load_decisions(path)
    assert "line 2" in str(caught.value)
    assert "format='json'" in str(caught.value)


def test_json_array_and_nested_key(tmp_path: Path) -> None:
    flat = tmp_path / "a.json"
    flat.write_text(json.dumps([ROW]), encoding="utf-8")
    nested = tmp_path / "b.json"
    nested.write_text(json.dumps({"decisions": [ROW]}), encoding="utf-8")
    assert load_decisions(flat) == load_decisions(nested)


def test_json_object_without_a_list_explains(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"rows_maybe": 1}), encoding="utf-8")
    with pytest.raises(DataError, match="not a list of decisions"):
        load_decisions(path)


def test_json_row_missing_state_lists_the_fields(tmp_path: Path) -> None:
    broken = {k: v for k, v in ROW.items() if k != "state"}
    with pytest.raises(DataError) as caught:
        load_decisions(_jsonl(tmp_path, broken))
    message = str(caught.value)
    assert "'state'" in message
    assert "question" in message  # the fields it does have
    assert "FieldMapping(state=" in message


def test_json_options_as_plain_strings_get_slug_ids(tmp_path: Path) -> None:
    row = dict(ROW, options=["Can wait a week", "Needs attention today"], answer_id="B")
    decisions = load_decisions(_jsonl(tmp_path, row))
    assert decisions[0].criterion.option_ids == ("can_wait_a_week", "needs_attention_today")
    assert decisions[0].answer_id == "needs_attention_today"


def test_json_single_option_is_refused(tmp_path: Path) -> None:
    row = dict(ROW, options=[{"id": "low", "description": "Can wait"}])
    with pytest.raises(DataError, match="at least two"):
        load_decisions(_jsonl(tmp_path, row))


# -------------------------------------------------------------------------- csv: long


LONG_CSV = (
    "id,state,question,options,option_ids,answer_id\n"
    "TKT-1,Login fails after reset,How urgent is this?,Can wait|Today,low|high,high\n"
    "TKT-2,Invoice is duplicated,How urgent is this?,Can wait|Today,low|high,low\n"
)


def test_csv_long_shape(tmp_path: Path) -> None:
    decisions = load_decisions(_csv(tmp_path, LONG_CSV))
    assert [d.id for d in decisions] == ["TKT-1", "TKT-2"]
    assert decisions[0].criterion.option_ids == ("low", "high")
    assert [d.answer_id for d in decisions] == ["high", "low"]
    assert decisions[0].criterion.key == "how_urgent_is_this"


def test_csv_long_autodetects_semicolon(tmp_path: Path) -> None:
    text = (
        "id,state,question,options\n"
        "TKT-1,Login fails,How urgent is this?,Can wait;Today\n"
    )
    decisions = load_decisions(_csv(tmp_path, text))
    assert decisions[0].criterion.option_ids == ("can_wait", "today")


def test_csv_long_explicit_delimiter(tmp_path: Path) -> None:
    text = "id,state,question,options\nTKT-1,Login fails,How urgent?,Can wait / Today\n"
    mapping = FieldMapping(option_delimiter="/")
    decisions = load_decisions(_csv(tmp_path, text), mapping=mapping)
    assert decisions[0].criterion.option_ids == ("can_wait", "today")


def test_csv_undelimited_options_cell_names_the_row(tmp_path: Path) -> None:
    text = "id,state,question,options\nTKT-7,Login fails,How urgent?,Today only\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "row 2" in message
    assert "'options'" in message
    assert "option_delimiter=" in message


def test_csv_option_ids_length_mismatch_names_both_columns(tmp_path: Path) -> None:
    text = (
        "id,state,question,options,option_ids\n"
        "TKT-3,Login fails,How urgent?,Can wait|Today|Now,low|high\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "id 'TKT-3'" in message
    assert "'option_ids'" in message
    assert "2 id(s)" in message and "3 option(s)" in message


def test_csv_empty_options_cell_names_the_row(tmp_path: Path) -> None:
    text = (
        "id,state,question,options\n"
        "TKT-1,Login fails,How urgent?,Can wait|Today\n"
        "TKT-2,Invoice dupe,How urgent?,\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    assert "id 'TKT-2'" in str(caught.value)


# -------------------------------------------------------------------------- csv: wide


WIDE_CSV = (
    "id,state,question,option_1,option_2,option_3,answer_id\n"
    "TKT-1,Login fails,How urgent?,Can wait,Today,Right now,Today\n"
    "TKT-2,Invoice dupe,How urgent?,Can wait,Today,,1\n"
)


def test_csv_wide_shape_with_ragged_tail(tmp_path: Path) -> None:
    decisions = load_decisions(_csv(tmp_path, WIDE_CSV))
    assert decisions[0].criterion.option_ids == ("can_wait", "today", "right_now")
    assert decisions[0].answer_id == "today"  # matched on description text
    assert decisions[1].criterion.option_ids == ("can_wait", "today")
    assert decisions[1].answer_id == "today"  # index 1


def test_csv_wide_with_explicit_ids(tmp_path: Path) -> None:
    text = (
        "id,state,question,option_1,option_1_id,option_2,option_2_id,answer_id\n"
        "TKT-1,Login fails,How urgent?,Can wait,low,Today,high,high\n"
    )
    decisions = load_decisions(_csv(tmp_path, text))
    assert decisions[0].criterion.option_ids == ("low", "high")
    assert decisions[0].answer_id == "high"


def test_csv_wide_hole_in_the_middle_is_refused(tmp_path: Path) -> None:
    text = (
        "id,state,question,option_1,option_2,option_3\n"
        "TKT-9,Login fails,How urgent?,Can wait,,Right now\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "id 'TKT-9'" in message
    assert "'option_2'" in message and "'option_3'" in message


def test_csv_wide_numbering_gap_is_refused(tmp_path: Path) -> None:
    text = "id,state,question,option_1,option_3\nTKT-1,Login fails,How urgent?,A,B\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    assert "option_2 is missing" in str(caught.value)


def test_csv_single_option_column_is_refused(tmp_path: Path) -> None:
    text = "id,state,question,option_1\nTKT-1,Login fails,How urgent?,A\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    assert "no second option column" in str(caught.value)


# ------------------------------------------------------------------- csv: shape sniff


def test_csv_ambiguous_shape_explains_both(tmp_path: Path) -> None:
    text = (
        "id,state,question,options,option_1,option_2\n"
        "TKT-1,Login fails,How urgent?,Can wait|Today,Can wait,Today\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "ambiguous" in message
    assert "(a) long" in message and "(b) wide" in message
    assert "option_1" in message and "option_2" in message


def test_csv_no_options_at_all_explains_both(tmp_path: Path) -> None:
    text = "id,state,question,answer_id\nTKT-1,Login fails,How urgent?,high\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "no options column" in message
    assert "(a) long" in message and "(b) wide" in message


# --------------------------------------------------------------------- csv: the mess


def test_csv_header_is_stripped_and_case_insensitive(tmp_path: Path) -> None:
    text = "﻿ ID , State ,QUESTION, Options \nTKT-1,Login fails,How urgent?,Can wait|Today\n"
    decisions = load_decisions(_csv(tmp_path, text))
    assert decisions[0].id == "TKT-1"
    assert decisions[0].state == "Login fails"


def test_csv_duplicate_header_is_refused(tmp_path: Path) -> None:
    text = "id,state,state,question,options\nTKT-1,a,b,How urgent?,Can wait|Today\n"
    with pytest.raises(DataError, match="repeats the column name"):
        load_decisions(_csv(tmp_path, text))


def test_csv_ragged_row_is_refused(tmp_path: Path) -> None:
    text = (
        "id,state,question,options\n"
        "TKT-1,Login fails,How urgent?,Can wait|Today\n"
        "TKT-2,Invoice, dupe,How urgent?,Can wait|Today\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "row 3" in message
    assert "quote any field" in message


def test_csv_missing_state_column_lists_what_it_has(tmp_path: Path) -> None:
    text = "id,evidence,question,options\nTKT-1,Login fails,How urgent?,Can wait|Today\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "no 'state' column" in message
    assert "evidence" in message
    assert "FieldMapping(state=" in message


def test_csv_blank_state_cell_names_the_row(tmp_path: Path) -> None:
    text = "id,state,question,options\nTKT-4,,How urgent?,Can wait|Today\n"
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    assert "id 'TKT-4'" in str(caught.value)
    assert "'state'" in str(caught.value)


def test_csv_header_only_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no data rows"):
        load_decisions(_csv(tmp_path, "id,state,question,options\n"))


def test_field_mapping_renames_everything(tmp_path: Path) -> None:
    text = (
        "ticket,summary,ask,choices,verdict,case\n"
        "T-1,Login fails,How urgent?,Can wait|Today,Today,case-1\n"
    )
    mapping = FieldMapping(
        state="summary",
        question="ask",
        options="choices",
        answer="verdict",
        group="case",
        id="ticket",
    )
    decisions = load_decisions(_csv(tmp_path, text), mapping=mapping)
    assert decisions[0].id == "T-1"
    assert decisions[0].grouping == "case-1"
    assert decisions[0].answer_id == "today"


def test_field_mapping_names_a_column_that_is_absent(tmp_path: Path) -> None:
    text = "id,state,question,options\nTKT-1,Login fails,How urgent?,Can wait|Today\n"
    with pytest.raises(DataError, match="no 'ticket' column"):
        load_decisions(_csv(tmp_path, text), mapping=FieldMapping(id="ticket"))


# ---------------------------------------------------------------------------- answers


@pytest.mark.parametrize("cell", ["high", "HIGH", "B", "1", "Today"])
def test_answer_cell_spellings_all_resolve(tmp_path: Path, cell: str) -> None:
    text = (
        "id,state,question,options,option_ids,answer_id\n"
        f"TKT-1,Login fails,How urgent?,Can wait|Today,low|high,{cell}\n"
    )
    assert load_decisions(_csv(tmp_path, text))[0].answer_id == "high"


@pytest.mark.parametrize("cell", ["", "N/A", "unknown"])
def test_blank_answer_cell_means_unlabelled(tmp_path: Path, cell: str) -> None:
    text = (
        "id,state,question,options,option_ids,answer_id\n"
        f"TKT-1,Login fails,How urgent?,Can wait|Today,low|high,{cell}\n"
    )
    assert load_decisions(_csv(tmp_path, text))[0].answer_id is None


def test_unknown_answer_names_row_column_and_permitted_ids(tmp_path: Path) -> None:
    text = (
        "id,state,question,options,option_ids,answer_id\n"
        "TKT-5,Login fails,How urgent?,Can wait|Today,low|high,critical\n"
    )
    with pytest.raises(DataError) as caught:
        load_decisions(_csv(tmp_path, text))
    message = str(caught.value)
    assert "id 'TKT-5'" in message
    assert "'answer_id'" in message
    assert "'critical'" in message
    assert "low, high" in message
    assert "A-B" in message


# ----------------------------------------------------------------- format resolution


def test_format_inferred_from_suffix_can_be_overridden(tmp_path: Path) -> None:
    path = tmp_path / "tickets.txt"
    path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="Cannot infer a format"):
        load_decisions(path)
    assert load_decisions(path, format="jsonl")[0].id == "tkt-1"


def test_unknown_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Unsupported format"):
        load_decisions(_jsonl(tmp_path, ROW), format="xlsx")


def test_missing_file_names_the_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(DataError) as caught:
        load_decisions(tmp_path / "nope.jsonl")
    assert str(tmp_path) in str(caught.value)


def test_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data.csv").mkdir()
    with pytest.raises(DataError, match="is a directory"):
        load_decisions(tmp_path / "data.csv")


def test_bare_name_is_neither_file_nor_dataset_id() -> None:
    with pytest.raises(DataError, match="neither an existing file nor"):
        load_decisions("tickets")


# ------------------------------------------------------------------- optional extras


def test_parquet_without_pyarrow_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    path = tmp_path / "tickets.parquet"
    path.write_bytes(b"PAR1")
    with pytest.raises(DataError) as caught:
        load_decisions(path)
    assert "pip install typedecide[parquet]" in str(caught.value)


def test_hf_dataset_without_datasets_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(DataError) as caught:
        load_decisions("acme/tickets")
    message = str(caught.value)
    assert "pip install typedecide[train]" in message
    assert "acme/tickets" in message
