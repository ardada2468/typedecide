"""Regression tests for bugs found in the data layer during the hardening audit.

Each test names the failure it pins. Most of them used to pass silently with the wrong
answer, which is the kind of bug a metrics table never shows.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide.data import loaders
from typedecide.data.loaders import FieldMapping, load_decisions, write_decisions
from typedecide.errors import DataError
from typedecide.schema import Criterion, Decision, Option

HEADER = "id,state,question,options,option_ids,answer_id\n"


def _write(tmp_path: Path, text: str, name: str = "tickets.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _row(ticket: str, answer: str, options: str = "Low|Mid|High", ids: str = "lo|mid|hi") -> str:
    return f"{ticket},State of {ticket},How urgent?,{options},{ids},{answer}\n"


# ------------------------------------------------------------- numeric answer columns


def test_one_based_answer_column_is_read_consistently(tmp_path: Path) -> None:
    """Row by row, "1" and "2" were read 0-based and "3" 1-based: two of three labels
    silently shifted one option along. The column has one convention; "3" proves it."""
    path = _write(tmp_path, HEADER + _row("t1", "1") + _row("t2", "2") + _row("t3", "3"))
    assert [d.answer_id for d in load_decisions(path)] == ["lo", "mid", "hi"]


def test_zero_based_answer_column_is_unchanged(tmp_path: Path) -> None:
    path = _write(tmp_path, HEADER + _row("t1", "0") + _row("t2", "1") + _row("t3", "2"))
    assert [d.answer_id for d in load_decisions(path)] == ["lo", "mid", "hi"]


def test_mixed_zero_and_one_based_answers_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, HEADER + _row("t1", "0") + _row("t2", "3"))
    with pytest.raises(DataError, match="mixes 0-based and 1-based") as caught:
        load_decisions(path)
    assert "'t1'" in str(caught.value) and "'t2'" in str(caught.value)


def test_ambiguous_numeric_answers_warn_and_read_zero_based(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path, HEADER + _row("t1", "1") + _row("t2", "2"))
    with caplog.at_level(logging.WARNING, logger="typedecide.data.loaders"):
        decisions = load_decisions(path)
    assert [d.answer_id for d in decisions] == ["mid", "hi"]
    assert "could be 0-based or 1-based" in caplog.text


def test_one_based_rewrite_survives_blank_jsonl_lines(tmp_path: Path) -> None:
    """The rewrite indexes the decisions list, not the physical line number."""
    options = ["Low", "Mid", "High"]
    rows = [
        {"id": f"t{n}", "state": f"s{n}", "question": "How urgent?", "options": options,
         "answer": str(n)}
        for n in (1, 2, 3)
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [d.answer_id for d in load_decisions(path)] == ["low", "mid", "high"]


@pytest.mark.parametrize("field", ["label", "answer_index"])
def test_positional_fields_are_never_read_as_one_based(tmp_path: Path, field: str) -> None:
    """`label: 2` over two options used to land on the second option. It is a 0-based
    position by definition, so 2 is out of range and must say so."""
    row = {"id": "t1", "state": "s", "question": "q?", "options": ["No", "Yes"], field: 2}
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="0-based position"):
        load_decisions(path)


def test_positional_label_in_range_still_loads(tmp_path: Path) -> None:
    row = {"id": "t1", "state": "s", "question": "q?", "options": ["No", "Yes"], "label": 1}
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_decisions(path)[0].answer_id == "yes"


def test_non_ascii_digit_answer_is_a_data_error_not_a_value_error(tmp_path: Path) -> None:
    """'²'.isdigit() is True and int('²') raises ValueError."""
    path = _write(tmp_path, HEADER + _row("t1", "²"))
    with pytest.raises(DataError, match="not one of this row's options"):
        load_decisions(path)


def test_float_spelling_of_an_index_resolves(tmp_path: Path) -> None:
    """pandas writes an integer column that has a blank in it as 0.0, 1.0, ..."""
    path = _write(tmp_path, HEADER + _row("t1", "0.0") + _row("t2", "2.0") + _row("t3", ""))
    assert [d.answer_id for d in load_decisions(path)] == ["lo", "hi", None]


def test_legacy_per_row_resolution_without_a_tracker() -> None:
    options = [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
    assert loaders._resolve_answer("2", options, "row 1", "answer") == "b"
    assert loaders._resolve_answer(1.0, options, "row 1", "answer") == "b"


# --------------------------------------------------------------------- invented ids


def test_long_options_sharing_a_prefix_get_distinct_ids(tmp_path: Path) -> None:
    """Suffix-then-truncate cut the suffix off a 64-character slug and handed back the
    id that had just collided, so the row failed as 'lists option id ... twice'."""
    stem = "the customer is asking for a refund because the product arrived damaged and "
    row = {"id": "t1", "state": "s", "question": "q?",
           "options": [stem + "wants money back", stem + "wants a replacement"]}
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ids = load_decisions(path)[0].criterion.option_ids
    assert len(set(ids)) == 2
    assert all(len(i) <= 64 for i in ids)
    assert ids[1].endswith("_2")


def test_invented_suffix_that_collides_is_bumped() -> None:
    paired = loaders._pair(["a 3", "a", "a"], None, "row 1")
    assert [o["id"] for o in paired] == ["a_3", "a", "a_4"]


# --------------------------------------------------------------------------- csv


def test_rows_using_different_option_delimiters_are_diagnosed(tmp_path: Path) -> None:
    """Every cell split on *something*, so the 'find the unsplittable cell' search came
    up empty and a bare StopIteration escaped."""
    text = (
        "id,state,question,options\n"
        "t1,s1,q?,Low|High\n"
        "t2,s2,q?,Low;High\n"
    )
    with pytest.raises(DataError, match="does not use one delimiter throughout") as caught:
        load_decisions(_write(tmp_path, text))
    assert "row 2" in str(caught.value) and "row 3" in str(caught.value)


def test_oversized_csv_field_is_a_data_error(tmp_path: Path) -> None:
    """csv.Error ('field larger than field limit') used to escape as a traceback."""
    huge = "x" * (csv.field_size_limit() + 10)
    text = f'id,state,question,options\nt1,"{huge}",q?,Low|High\n'
    with pytest.raises(DataError, match="CSV parser gave up"):
        load_decisions(_write(tmp_path, text))


@pytest.mark.parametrize("name", ["tickets.csv", "tickets.jsonl", "tickets.json"])
def test_non_utf8_file_is_a_data_error(tmp_path: Path, name: str) -> None:
    """A Windows-1252 export from Excel used to raise UnicodeDecodeError."""
    path = tmp_path / name
    path.write_bytes("id,state,question,options\nt1,caf\xe9 closed,q?,Low|High\n".encode("cp1252"))
    with pytest.raises(DataError, match="is not UTF-8") as caught:
        load_decisions(path)
    assert "0xe9" in str(caught.value)


def test_quoted_newlines_and_crlf_survive(tmp_path: Path) -> None:
    text = (
        "id,state,question,options,answer_id\r\n"
        't1,"line one\r\nline two",q?,Low|High,High\r\n'
        "t2,plain,q?,Low|High,Low\r\n"
    )
    decisions = load_decisions(_write(tmp_path, text))
    assert decisions[0].state == "line one\r\nline two"
    assert [d.answer_id for d in decisions] == ["high", "low"]


def test_row_numbers_count_records_not_physical_lines(tmp_path: Path) -> None:
    text = (
        "id,state,question,options\n"
        't1,"two\nlines",q?,Low|High\n'
        "t2,,q?,Low|High\n"
    )
    with pytest.raises(DataError, match="row 3"):
        load_decisions(_write(tmp_path, text))


def test_csv_is_streamed_not_slurped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, HEADER + _row("t1", "lo") + _row("t2", "hi"))

    def refuse(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("the CSV loader read the whole file into memory")

    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert len(load_decisions(path)) == 2


def test_whitespace_only_csv_is_empty(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="is empty"):
        load_decisions(_write(tmp_path, "\n  \n\n"))


def test_blank_header_line_is_diagnosed(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="has no ','"):
        load_decisions(_write(tmp_path, "\nid,state\n"))
    with pytest.raises(DataError, match="blank first line"):
        load_decisions(_write(tmp_path, ",,\nt1,s,q\n"))


def test_tsv_is_read_with_tabs(tmp_path: Path) -> None:
    text = "id\tstate\tquestion\toptions\tanswer_id\nt1\ts, with comma\tq?\tLow|High\tB\n"
    decision = load_decisions(_write(tmp_path, text, "tickets.tsv"))[0]
    assert decision.state == "s, with comma" and decision.answer_id == "high"


def test_every_options_cell_empty_is_diagnosed(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="every cell of column 'options' is empty"):
        load_decisions(_write(tmp_path, "id,state,question,options\nt1,s,q?,\n"))


def test_unreadable_file_is_a_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, HEADER + _row("t1", "lo"))

    def deny(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(DataError, match="Could not read"):
        load_decisions(path)


# --------------------------------------------------------------------- format sniff


def test_directory_and_missing_and_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="is a directory"):
        load_decisions(tmp_path, format="jsonl")
    with pytest.raises(DataError, match="No such dataset file"):
        load_decisions(tmp_path / "absent.jsonl")
    odd = tmp_path / "tickets.dat"
    odd.write_text("{}", encoding="utf-8")
    with pytest.raises(DataError, match="Cannot infer a format"):
        load_decisions(odd)
    with pytest.raises(DataError, match="Unsupported format"):
        load_decisions(odd, format="xml")
    with pytest.raises(DataError, match="neither an existing file nor"):
        load_decisions("not a dataset id")


def test_format_aliases(tmp_path: Path) -> None:
    path = tmp_path / "tickets.dat"
    row = {"id": "t1", "state": "s", "question": "q?", "options": ["No", "Yes"]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(load_decisions(path, format="ndjson")) == 1
    assert len(load_decisions(path, format=".JSONL")) == 1


# ----------------------------------------------------------------- hugging face


class _FakeDatasets(types.ModuleType):
    def __init__(self, rows: list[dict[str, Any]], error: Exception | None = None) -> None:
        super().__init__("datasets")
        self.rows = rows
        self.error = error
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def load_dataset(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.rows


HF_ROW = {"id": "h1", "state": "s", "question": "q?", "options": ["No", "Yes"], "answer": "Yes"}


def test_hf_id_with_a_split_is_inferred_and_never_trusts_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error message recommended 'owner/name:validation', which the id pattern then
    refused. And loading data must never opt in to running the dataset's own script."""
    fake = _FakeDatasets([HF_ROW])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    decisions = load_decisions("owner/name:validation")
    assert decisions[0].answer_id == "yes"
    (args, kwargs), = fake.calls
    assert args == ("owner/name",) and kwargs == {"split": "validation"}
    assert "trust_remote_code" not in kwargs


def test_hf_default_split_and_empty_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDatasets([HF_ROW])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    load_decisions("owner/name")
    assert fake.calls[-1][1] == {"split": "train"}

    monkeypatch.setitem(sys.modules, "datasets", _FakeDatasets([]))
    with pytest.raises(DataError, match="yielded no rows"):
        load_decisions("owner/name")

    broken = _FakeDatasets([], error=ConnectionError("offline"))
    monkeypatch.setitem(sys.modules, "datasets", broken)
    with pytest.raises(DataError, match="never enables trust_remote_code") as caught:
        load_decisions("owner/name", format="hf")
    assert "offline" in str(caught.value)


# ----------------------------------------------------------------------- parquet


def test_parquet_round_trip_in_batches(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        {"id": f"p{n}", "state": f"state {n}", "question": "q?", "options": ["No", "Yes"],
         "answer": "Yes" if n % 2 else "No", "meta_source": "unit"}
        for n in range(7)
    ]
    path = tmp_path / "tickets.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    decisions = load_decisions(path)
    assert [d.id for d in decisions] == [f"p{n}" for n in range(7)]
    assert decisions[1].answer_id == "yes" and decisions[1].meta == {"meta_source": "unit"}


def test_corrupt_parquet_is_a_data_error(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "tickets.parquet"
    path.write_bytes(b"id,state\nthis is a csv\n")
    with pytest.raises(DataError, match="could not be read as Parquet"):
        load_decisions(path)


def test_empty_parquet_is_a_data_error(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "tickets.parquet"
    pq.write_table(pa.table({"state": pa.array([], pa.string())}), path)
    with pytest.raises(DataError, match="contained no rows"):
        load_decisions(path)


# ------------------------------------------------------------------------- writing


def _decision(ident: str = "d1", **meta: Any) -> Decision:
    criterion = Criterion("urgency", "How urgent?", (Option("lo", "Low"), Option("hi", "High")))
    return Decision(id=ident, state="café closed", criterion=criterion, answer_id="hi",
                    meta=meta)


def test_write_is_atomic_and_leaves_no_scratch_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.jsonl"
    write_decisions([_decision("a"), _decision("b")], target)
    assert sorted(p.name for p in target.parent.iterdir()) == ["out.jsonl"]
    assert [d.id for d in load_decisions(target)] == ["a", "b"]


def test_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    """A crash mid-write used to leave a truncated dataset that still parsed."""
    target = tmp_path / "out.jsonl"
    write_decisions([_decision("kept")], target)

    def explode() -> Any:
        yield _decision("new")
        raise OSError(28, "No space left on device")

    class Rows(list):  # type: ignore[type-arg]
        def __iter__(self) -> Any:
            return explode()

    with pytest.raises(DataError, match="Could not write"):
        write_decisions(Rows([_decision("new")]), target)
    assert [d.id for d in load_decisions(target)] == ["kept"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl"]


def test_written_bytes_use_lf_and_utf8_on_every_platform(tmp_path: Path) -> None:
    for name in ("out.jsonl", "out.json"):
        target = tmp_path / name
        write_decisions([_decision("a"), _decision("b")], target)
        raw = target.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        assert "café".encode() in raw


def test_empty_write_is_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    write_decisions([], target)
    assert target.read_bytes() == b""


def test_unserialisable_meta_does_not_crash_a_write(tmp_path: Path) -> None:
    """Parquet and HF rows carry datetimes; json.dumps used to raise TypeError."""
    import datetime

    target = tmp_path / "out.jsonl"
    write_decisions([_decision("a", meta_seen=datetime.date(2026, 1, 2))], target)
    assert json.loads(target.read_text(encoding="utf-8"))["meta_seen"] == "2026-01-02"


# ------------------------------------------------------------- json row edge cases


def test_json_row_shapes_are_diagnosed(tmp_path: Path) -> None:
    cases: list[tuple[Any, str]] = [
        ([1, 2], "item 1 is a int"),
        ("just a string", "holds a str"),
        ({"rows": []}, "contained no rows"),
        ([{"id": "t", "state": "s", "question": "q?"}], "no 'options' field"),
        ([{"id": "t", "state": "s", "question": "q?", "options": 5}], "holds a int"),
        ([{"id": "t", "state": "s", "question": "q?", "options": "[broken"}], "looks like JSON"),
        ([{"id": "t", "state": "s", "question": "", "options": ["a", "b"]}], "'question'"),
        ([{"id": "t", "state": "s", "question": "q?",
           "options": [{"id": "a"}, {"id": "b", "description": "B"}]}], "no 'description'"),
    ]
    for index, (payload, expected) in enumerate(cases):
        path = tmp_path / f"case{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(DataError, match=expected):
            load_decisions(path)


def test_json_options_as_strings_and_mapping(tmp_path: Path) -> None:
    rows = [
        {"ticket": "T-1", "body": "s1", "ask": "q?", "choices": '["No", "Yes"]', "gold": "B",
         "case": "c1", "meta_team": "billing"},
        {"ticket": "T-2", "body": "s2", "ask": "q?", "choices": "No; Yes", "gold": "No",
         "case": "c2"},
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"data": rows}), encoding="utf-8")
    mapping = FieldMapping(state="body", question="ask", options="choices", answer="gold",
                           group="case", id="ticket")
    first, second = load_decisions(path, mapping=mapping)
    assert (first.id, first.answer_id, first.group_id) == ("T-1", "yes", "c1")
    assert first.meta == {"meta_team": "billing"}
    assert second.answer_id == "no"


def test_jsonl_line_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DataError, match="holds a list, not an object"):
        load_decisions(path)
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DataError, match="contained no rows"):
        load_decisions(path)


def test_mapped_id_missing_from_a_json_row(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(HF_ROW) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="no 'ticket' field"):
        load_decisions(path, mapping=FieldMapping(id="ticket"))


def test_duplicate_explicit_option_ids_are_refused(tmp_path: Path) -> None:
    text = "id,state,question,options,option_ids\nt1,s,q?,Low|High,x|x\n"
    with pytest.raises(DataError, match="appears twice"):
        load_decisions(_write(tmp_path, text))


# ------------------------------------------------------------ remaining refusals


def test_remaining_csv_refusals_name_the_row_and_column(tmp_path: Path) -> None:
    cases = [
        ("id,state,question,options\nt1,s,,Low|High\n", "column 'question' is empty"),
        ("id,state,question,options\nt1,s,q?,Low|High\nt2,s,q?,Only\n",
         "none of the delimiters"),
        ("id,state,question,option_1,option_2\nt1,s,q?,Only,\n", "only 1 of the option columns"),
        ("id,state,question,options\nt1,s,q?," + "|".join(f"o{i}" for i in range(21)) + "\n",
         "row 2, id 't1': Criterion .* has 21 options"),
    ]
    for index, (text, message) in enumerate(cases):
        with pytest.raises(DataError, match=message):
            load_decisions(_write(tmp_path, text, f"case{index}.csv"))

    text = "id,state,question,options\nt1,s,q?,Low|High\n"
    with pytest.raises(DataError, match="no 'Case' column for the group"):
        load_decisions(_write(tmp_path, text), mapping=FieldMapping(group="Case"))
    with pytest.raises(DataError, match="splits on ';' into 1 option"):
        load_decisions(_write(tmp_path, text), mapping=FieldMapping(option_delimiter=";"))


def test_options_with_no_sluggable_text_get_positional_ids(tmp_path: Path) -> None:
    text = "id,state,question,options,answer_id\nt1,s,q?,不|是,是\n"
    decision = load_decisions(_write(tmp_path, text))[0]
    assert decision.criterion.option_ids == ("option_1", "option_2")
    assert decision.answer_id == "option_2"


def test_answer_given_as_a_sluggable_spelling_of_the_id(tmp_path: Path) -> None:
    text = HEADER + "t1,s,q?,Can wait|Right now,can_wait|right_now,Right-Now\n"
    assert load_decisions(_write(tmp_path, text))[0].answer_id == "right_now"


def test_invalid_json_document_is_diagnosed(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('[{"id": 1},]', encoding="utf-8")
    with pytest.raises(DataError, match="is not valid JSON .* use format='jsonl'"):
        load_decisions(path)


# ------------------------------------------- blank-words that are really answers


@pytest.mark.parametrize(
    ("options", "ids", "cell", "expected"),
    [
        ("Billing|Cannot tell", "billing|unknown", "unknown", "unknown"),   # an option id
        ("Billing|Cannot tell", "billing|unknown", "UNKNOWN", "unknown"),   # ... any case
        ("Billing|Unknown", "", "Unknown", "unknown"),                      # a description
        ("Applies|N/A", "yes|n_a", "N/A", "n_a"),                           # slugs to an id
        ("Some|None", "some|none", "none", "none"),
        ("Billing|Shipping", "billing|shipping", "unknown", None),          # not an option
        ("Billing|Shipping", "billing|shipping", "N/A", None),
        ("Billing|Shipping", "billing|shipping", "-", None),
        ("Billing|Shipping", "billing|shipping", "", None),
    ],
)
def test_a_blank_word_that_is_one_of_the_rows_options_is_a_label(
    tmp_path: Path, options: str, ids: str, cell: str, expected: str | None
) -> None:
    """The blank-word check ran before option matching, so a row whose gold was the
    option `unknown` was read as unlabelled -- and then dropped from every score."""
    text = HEADER + f"t1,Evidence,Which queue?,{options},{ids},{cell}\n"
    assert load_decisions(_write(tmp_path, text))[0].answer_id == expected


def test_a_blank_word_answer_in_json_rows(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "state": "s", "question": "q?", "answer": "unknown",
         "options": [{"id": "known", "description": "Known"},
                     {"id": "unknown", "description": "Cannot be determined"}]},
        {"id": "b", "state": "s", "question": "q?", "answer": "unknown",
         "options": ["Yes", "No"]},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [d.answer_id for d in load_decisions(path)] == ["unknown", None]


def test_a_mis_mapped_answer_column_does_not_quote_the_evidence(tmp_path: Path) -> None:
    """Point --col-answer at the ticket body by mistake and the whole state used to be
    reproduced in the error message, and from there in logs and bug reports."""
    body = "Customer SSN 000-00-0000 and card ending 4242. " * 20
    text = f'id,state,question,options,gold\nt1,"{body}",q?,Low|High,high\n'
    mapping = FieldMapping(answer="state")
    with pytest.raises(DataError, match="not one of this row's") as caught:
        load_decisions(_write(tmp_path, text), mapping=mapping)
    assert len(str(caught.value)) < 500 and body.strip() not in str(caught.value)
    assert "..." in str(caught.value)
