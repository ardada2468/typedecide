"""Reading decisions out of whatever the customer already has.

This is the first surface a new user touches, and the data they bring is a CSV a
ticketing system exported at four in the afternoon. So every failure here names the
row, names the column, quotes the offending value and says what would fix it. A
loader that says "invalid input" has made the library someone else's problem.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn, TypeAlias

from ..errors import DataError, SchemaError
from ..schema import Decision, decision_from_dict, decision_to_dict

LOGGER = logging.getLogger("typedecide.data.loaders")

__all__ = ["FieldMapping", "load_decisions", "write_decisions"]

#: Suffix -> format name. Anything else has to be named explicitly.
SUFFIX_FORMATS: dict[str, str] = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}

FORMATS: tuple[str, ...] = ("jsonl", "json", "csv", "parquet", "hf")

#: Tried in this order when the options column's delimiter is not given. Two-character
#: candidates come first so "a||b" is not read as three empty-ish fields.
OPTION_DELIMITERS: tuple[str, ...] = ("||", "|", ";", "\t", ",")

_WIDE_OPTION = re.compile(r"^option[ _-]?(\d+)$", re.IGNORECASE)
_WIDE_OPTION_ID = re.compile(r"^option[ _-]?(\d+)[ _-]?id$", re.IGNORECASE)
# 'owner/name', optionally followed by ':split' -- e.g. 'owner/name:validation'.
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*(/[\w.-]+)?(:[\w.\[\]:%+-]+)?$")
_WHOLE_NUMBER = re.compile(r"^(\d+)(?:\.0+)?$", re.ASCII)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_BLANK_WORDS = frozenset({"", "na", "n/a", "nan", "none", "null", "-", "?", "unknown"})

#: Resolves a wanted column name onto the file's actual spelling, case-insensitively.
Resolver: TypeAlias = Callable[[str | None], str | None]

_SHAPES_HELP = (
    "A CSV expresses its options in one of two shapes: (a) long -- one 'options' column "
    "holding a delimited list of descriptions, optionally beside an 'option_ids' column "
    "holding the matching ids in the same order; or (b) wide -- columns 'option_1', "
    "'option_2', ... 'option_n' holding one description each, optionally beside "
    "'option_1_id', 'option_2_id', ... for the ids."
)


@dataclass(frozen=True)
class FieldMapping:
    """Maps a customer's column names onto our schema, so bring-your-own-data does
    not mean rename-your-columns-first.

    The first six fields are the contract. The rest exist because CSV has to express a
    set of options in a flat table, and there is no one spelling of that. ``group`` and
    ``id`` default to ``None``, which means "use the conventional column if the file
    happens to have one, otherwise leave it unset".
    """

    state: str = "state"
    question: str = "question"
    options: str = "options"
    answer: str = "answer_id"
    group: str | None = None
    id: str | None = None
    #: Column holding the option ids, parallel to ``options`` (long CSV shape only).
    option_ids: str = "option_ids"
    #: Column holding the criterion key. Derived from the question when absent.
    key: str = "key"
    #: Delimiter inside ``options``/``option_ids``. ``None`` auto-detects.
    option_delimiter: str | None = None


# --------------------------------------------------------------------------- public


def load_decisions(
    source: str | Path,
    *,
    format: str | None = None,  # noqa: A002 - the contract names this argument
    mapping: FieldMapping | None = None,
) -> list[Decision]:
    """Load decisions from a file or a Hugging Face dataset id.

    ``format`` is inferred from the suffix when not given: ``.jsonl``/``.ndjson``,
    ``.json``, ``.csv``/``.tsv``, ``.parquet``/``.pq``. A string that is not an
    existing path and looks like ``owner/name`` is read as a HF dataset id.
    """
    mapping = mapping or FieldMapping()
    resolved = _resolve_format(source, format)

    if resolved == "hf":
        return _load_hf(str(source), mapping)

    path = Path(source)
    if not path.exists():
        raise DataError(
            f"No such dataset file: {path.resolve()}. "
            "Fix: check the path, or pass a Hugging Face dataset id such as 'owner/name'."
        )
    if path.is_dir():
        raise DataError(
            f"{path.resolve()} is a directory, not a dataset file. "
            "Fix: name the .jsonl, .json, .csv or .parquet file inside it."
        )

    loaders: dict[str, Callable[[Path, FieldMapping], list[Decision]]] = {
        "jsonl": _load_jsonl,
        "json": _load_json,
        "csv": _load_csv,
        "parquet": _load_parquet,
    }
    loader = loaders.get(resolved)
    if loader is None:  # pragma: no cover - _resolve_format only returns names in FORMATS
        raise DataError(f"Unknown format {resolved!r}. Fix: pass one of {', '.join(FORMATS)}.")
    try:
        return loader(path, mapping)
    except UnicodeDecodeError as error:
        raise DataError(
            f"{path.name} is not UTF-8: byte 0x{error.object[error.start]:02x} at offset "
            f"{error.start} does not decode. Fix: re-export the file as UTF-8 (in Excel, "
            "'CSV UTF-8 (Comma delimited)'), or convert it with "
            "`iconv -f WINDOWS-1252 -t UTF-8`."
        ) from error
    except OSError as error:
        raise DataError(
            f"Could not read {path}: {error}. Fix: check the file's permissions."
        ) from error


def write_decisions(decisions: Sequence[Decision], path: Path) -> None:
    """Write decisions back out. ``.json`` gets an array, anything else gets JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Written beside the target and renamed over it, so a crash or a full disk leaves
    # the previous file intact rather than a truncated dataset that still parses.
    # newline="\n" keeps the bytes identical on Windows, where text mode would
    # otherwise write "\r\n" and change the file's hash from one machine to the next.
    scratch = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with scratch.open("w", encoding="utf-8", newline="\n") as handle:
            if path.suffix.lower() == ".json":
                payloads = [decision_to_dict(d) for d in decisions]
                handle.write(json.dumps(payloads, ensure_ascii=False, indent=2, default=str))
                handle.write("\n")
            else:
                # One row at a time: JSONL never needs the whole dataset serialised at once.
                for decision in decisions:
                    handle.write(
                        json.dumps(decision_to_dict(decision), ensure_ascii=False, default=str)
                    )
                    handle.write("\n")
        scratch.replace(path)
    except OSError as error:
        raise DataError(f"Could not write {path}: {error}.") from error
    finally:
        with contextlib.suppress(OSError):
            scratch.unlink()
    LOGGER.info("wrote %d decisions to %s", len(decisions), path)


# ---------------------------------------------------------------------- format sniff


def _resolve_format(source: str | Path, format: str | None) -> str:  # noqa: A002
    if format is not None:
        name = format.lower().lstrip(".")
        alias = {"ndjson": "jsonl", "tsv": "csv", "pq": "parquet", "huggingface": "hf"}
        name = alias.get(name, name)
        if name not in FORMATS:
            raise DataError(
                f"Unsupported format {format!r}. Fix: pass one of {', '.join(FORMATS)}."
            )
        return name

    text = str(source)
    suffix = Path(text).suffix.lower()
    if suffix in SUFFIX_FORMATS:
        return SUFFIX_FORMATS[suffix]
    if Path(text).exists():
        raise DataError(
            f"Cannot infer a format from {text!r}: the suffix {suffix or '(none)'} is not one of "
            f"{', '.join(sorted(SUFFIX_FORMATS))}. "
            f"Fix: pass format='jsonl' (or {', '.join(FORMATS[1:])}) explicitly."
        )
    if _HF_ID.match(text) and "/" in text:
        return "hf"
    raise DataError(
        f"{text!r} is neither an existing file nor a Hugging Face dataset id. "
        f"Fix: give a path ending in one of {', '.join(sorted(SUFFIX_FORMATS))}, or a dataset "
        "id of the form 'owner/name'."
    )


# ----------------------------------------------------------------------- json family


def _load_jsonl(path: Path, mapping: FieldMapping) -> list[Decision]:
    decisions: list[Decision] = []
    label = path.name
    numeric = _IndexedAnswers()
    # Streamed a line at a time: a JSONL export is never held in memory as text.
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(
                    f"{label} line {line_no} is not valid JSON ({error.msg} at column "
                    f"{error.colno}). Fix: one complete JSON object per line, no trailing "
                    "commas; use format='json' if the file is a single JSON array."
                ) from error
            if not isinstance(payload, dict):
                raise DataError(
                    f"{label} line {line_no} holds a {type(payload).__name__}, not an object. "
                    "Fix: each line must be a JSON object describing one decision."
                )
            decisions.append(
                _decision_from_mapping(
                    payload, mapping, label, line_no, numeric, position=len(decisions)
                )
            )
    if not decisions:
        raise DataError(f"{path} contained no rows. Fix: check the file is not empty.")
    numeric.settle(decisions, label)
    LOGGER.info("loaded %d decisions from %s", len(decisions), path)
    return decisions


def _load_json(path: Path, mapping: FieldMapping) -> list[Decision]:
    # A single JSON document cannot be parsed incrementally with the standard library,
    # so this is the one text format read whole. Use JSONL for anything large.
    label = path.name
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise DataError(
            f"{label} is not valid JSON ({error.msg} at line {error.lineno}, column "
            f"{error.colno}). Fix: use format='jsonl' if this file is one object per line."
        ) from error

    rows: Any = payload
    if isinstance(payload, dict):
        for key in ("decisions", "data", "rows", "examples"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        else:
            raise DataError(
                f"{label} holds a JSON object with keys {sorted(payload)[:8]}, not a list of "
                "decisions. Fix: make the file a JSON array, or nest the array under "
                "'decisions'."
            )
    if not isinstance(rows, list):
        raise DataError(
            f"{label} holds a {type(rows).__name__}, not a list of decisions. "
            "Fix: make the top level a JSON array of objects."
        )

    decisions: list[Decision] = []
    numeric = _IndexedAnswers()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise DataError(
                f"{label} item {index} is a {type(row).__name__}, not an object. "
                "Fix: every item in the array must be an object describing one decision."
            )
        decisions.append(
            _decision_from_mapping(row, mapping, label, index, numeric, position=len(decisions))
        )
    if not decisions:
        raise DataError(f"{path} contained no rows. Fix: check the file is not empty.")
    numeric.settle(decisions, label)
    LOGGER.info("loaded %d decisions from %s", len(decisions), path)
    return decisions


def _load_parquet(path: Path, mapping: FieldMapping) -> list[Decision]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise DataError(
            "Reading Parquet needs pyarrow, which is an optional extra. "
            "Fix: pip install typedecide[parquet]"
        ) from error

    label = path.name
    decisions: list[Decision] = []
    numeric = _IndexedAnswers()
    index = 0
    try:
        # Record batches, not read_table(): only one batch of rows exists as Python
        # objects at a time, instead of the whole file twice over.
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                index += 1
                decisions.append(
                    _decision_from_mapping(
                        row, mapping, label, index, numeric, position=len(decisions)
                    )
                )
    except pa.ArrowException as error:
        raise DataError(
            f"{label} could not be read as Parquet: {error}. Fix: check the file is a "
            "complete Parquet file and not, say, a CSV with the wrong suffix."
        ) from error
    if not decisions:
        raise DataError(f"{path} contained no rows. Fix: check the file is not empty.")
    numeric.settle(decisions, label)
    LOGGER.info("loaded %d decisions from %s", len(decisions), path)
    return decisions


def _load_hf(dataset_id: str, mapping: FieldMapping) -> list[Decision]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise DataError(
            f"Reading the Hugging Face dataset {dataset_id!r} needs the 'datasets' package, "
            "which is an optional extra. Fix: pip install typedecide[train]"
        ) from error

    name, _, split = dataset_id.partition(":")
    # `trust_remote_code` is deliberately never passed, so it stays at the library's
    # default of False: a dataset id is data, and loading data must not run the code
    # that happens to be published beside it. A dataset that needs a loading script is
    # refused by `datasets` itself, and that refusal is reported below.
    try:
        loaded = load_dataset(name, split=split or "train")
    except Exception as error:
        raise DataError(
            f"Could not load the Hugging Face dataset {name!r} (split {split or 'train'!r}): "
            f"{error}. Fix: check the id, the split name and your network access. typedecide "
            "never enables trust_remote_code, so a dataset that needs a loading script must "
            "be exported to JSONL or Parquet first."
        ) from error
    numeric = _IndexedAnswers()
    decisions: list[Decision] = []
    for index, row in enumerate(loaded, start=1):
        decisions.append(
            _decision_from_mapping(
                dict(row), mapping, dataset_id, index, numeric, position=len(decisions)
            )
        )
    if not decisions:
        raise DataError(
            f"Hugging Face dataset {dataset_id!r} yielded no rows. "
            "Fix: check the split name, e.g. 'owner/name:validation'."
        )
    numeric.settle(decisions, dataset_id)
    LOGGER.info("loaded %d decisions from %s", len(decisions), dataset_id)
    return decisions


# ------------------------------------------------------------------------------- csv


def _load_csv(path: Path, mapping: FieldMapping) -> list[Decision]:
    """Two streaming passes over the file, neither of which holds it in memory.

    The first counts the rows and works out which delimiter splits *every* options
    cell -- a whole-column question. The second builds the decisions. Reading twice is
    the price of not keeping the text, the parsed grid and the row dicts alive at once,
    which on a large export is three copies of the file.
    """
    label = path.name
    first_line = _first_line(path)
    if first_line is None:
        raise DataError(f"{path} is empty. Fix: export the rows again, header included.")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(first_line, label)

    scan = _scan_csv(path, delimiter, label, mapping.options)
    header = scan.header
    if not any(header):
        raise DataError(
            f"{label} has a blank first line where the header should be. "
            "Fix: the first line must name the columns."
        )

    duplicates = sorted({c for c in header if header.count(c) > 1 and c})
    if duplicates:
        raise DataError(
            f"{label} header repeats the column name(s) {', '.join(duplicates)}. "
            "Fix: give every column a distinct name; the second copy would silently win."
        )

    if scan.records == 0:
        raise DataError(
            f"{label} has a header but no data rows. Fix: export the rows, not just the header."
        )

    columns = [c for c in header if c]
    resolve = _column_resolver(columns)
    state_col = _require_column(resolve, mapping.state, "state", label, columns)
    question_col = _require_column(resolve, mapping.question, "question", label, columns)
    answer_col = resolve(mapping.answer) or resolve("answer") or resolve("label")
    key_col = resolve(mapping.key) or resolve("criterion")
    id_col = resolve(mapping.id) if mapping.id else resolve("id")
    group_col = resolve(mapping.group) if mapping.group else resolve("group_id")
    if mapping.id and id_col is None:
        raise DataError(_missing_column(label, mapping.id, "id", columns))
    if mapping.group and group_col is None:
        raise DataError(_missing_column(label, mapping.group, "group", columns))

    shape, long_cols, wide_cols = _detect_option_shape(columns, mapping, resolve, label)
    delimiter_used = mapping.option_delimiter or ""
    if shape == "long" and not delimiter_used:
        delimiter_used = _choose_option_delimiter(scan, long_cols[0], label)

    decisions: list[Decision] = []
    numeric = _IndexedAnswers()
    for line_no, cells in _csv_records(path, delimiter, label):
        _check_width(cells, header, label, line_no, delimiter)
        padded = list(cells) + [""] * (len(header) - len(cells))
        row = {k: v for k, v in zip(header, padded, strict=True) if k}

        row_id = (row.get(id_col) or "").strip() if id_col else ""
        where = _where(label, line_no, row_id)
        state = (row.get(state_col) or "").strip()
        if not state:
            raise DataError(
                f"{where}: column {state_col!r} is empty. Fix: every decision needs the "
                "evidence text the model reads; drop the row or fill the cell."
            )
        question = (row.get(question_col) or "").strip()
        if not question:
            raise DataError(
                f"{where}: column {question_col!r} is empty. Fix: every decision needs the "
                "question being asked about that evidence."
            )

        if shape == "long":
            options = _options_from_long(row, long_cols, delimiter_used, where)
        else:
            options = _options_from_wide(row, wide_cols, where)

        payload: dict[str, Any] = {
            "state": state,
            "question": question,
            "options": options,
            "key": _criterion_key(row.get(key_col) if key_col else None, question),
        }
        if row_id:
            payload["id"] = row_id
        if group_col and (row.get(group_col) or "").strip():
            payload["group_id"] = row[group_col].strip()
        if answer_col:
            payload["answer_id"] = _resolve_answer(
                row.get(answer_col), options, where, answer_col,
                numeric=numeric, position=len(decisions),
            )
        decisions.append(_build(payload, where))

    numeric.settle(decisions, label)
    LOGGER.info("loaded %d decisions from %s (%s option shape)", len(decisions), path, shape)
    return decisions


def _first_line(path: Path) -> str | None:
    """The file's first physical line, or None if the file is nothing but whitespace."""
    first: str | None = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if first is None:
                first = line.rstrip("\r\n")
            if line.strip():
                return first
    return None


def _csv_records(path: Path, delimiter: str, label: str) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(row number, cells)`` for every non-blank data row, header excluded.

    Row numbers count records the way a spreadsheet does -- the header is row 1 -- so
    they match what the user sees when they open the file, even when a quoted field
    spans several physical lines.
    """
    yield from (
        (row_no, cells)
        for row_no, cells in _csv_rows(path, delimiter, label)
        if row_no > 1 and any(cell.strip() for cell in cells)
    )


def _csv_rows(path: Path, delimiter: str, label: str) -> Iterator[tuple[int, list[str]]]:
    # newline="" is what the csv module requires: it must see the raw line endings to
    # keep a newline inside a quoted field intact on every platform.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            yield from enumerate(reader, start=1)
        except csv.Error as error:
            raise DataError(
                f"{label}: the CSV parser gave up at line {reader.line_num} ({error}). Fix: "
                f"a field over {csv.field_size_limit():,} characters nearly always means an "
                "unbalanced quote has swallowed the rest of the file, so check the quoting "
                "on and before that line. If the field really is that long it is far past "
                "any model's context; shorten the evidence before loading."
            ) from error


@dataclass
class _CsvScan:
    """What the first pass learns: the header, the row count, and which option
    delimiters split every filled options cell."""

    header: list[str]
    records: int = 0
    filled: int = 0
    viable: list[str] = field(default_factory=lambda: list(OPTION_DELIMITERS))
    #: First options cell that no candidate delimiter splits, as (row, cell).
    unsplittable: tuple[int, str] | None = None
    #: Two rows that need different delimiters, kept for the error message.
    examples: list[tuple[int, str]] = field(default_factory=list)


def _scan_csv(path: Path, delimiter: str, label: str, options_name: str) -> _CsvScan:
    scan = _CsvScan(header=[])
    options_at: int | None = None
    for row_no, cells in _csv_rows(path, delimiter, label):
        if row_no == 1:
            scan.header = [cell.strip() for cell in cells]
            wanted = options_name.strip().casefold()
            options_at = next(
                (i for i, name in enumerate(scan.header) if name and name.casefold() == wanted),
                None,
            )
            continue
        if not any(cell.strip() for cell in cells):
            continue
        _check_width(cells, scan.header, label, row_no, delimiter)
        scan.records += 1
        if options_at is None or options_at >= len(cells):
            continue
        cell = cells[options_at].strip()
        if not cell:
            continue
        scan.filled += 1
        splits = [d for d in OPTION_DELIMITERS if _splits(cell, d)]
        if not splits and scan.unsplittable is None:
            scan.unsplittable = (row_no, cell)
        before = list(scan.viable)
        scan.viable = [d for d in scan.viable if d in splits]
        if len(scan.examples) < 2 and (not scan.examples or (before and not scan.viable)):
            scan.examples.append((row_no, cell))
    return scan


def _check_width(
    cells: Sequence[str], header: Sequence[str], label: str, row_no: int, delimiter: str
) -> None:
    if len(cells) > len(header):
        raise DataError(
            f"{label} row {row_no} has {len(cells)} fields but the header names "
            f"{len(header)}. Fix: quote any field containing a {delimiter!r}; the extra "
            f"field starts with {cells[len(header)][:40]!r}."
        )


def _splits(cell: str, delimiter: str) -> bool:
    return len([part for part in cell.split(delimiter) if part.strip()]) >= 2


def _sniff_delimiter(header: str, label: str) -> str:
    counts = {d: header.count(d) for d in (",", "\t", ";")}
    best = max(counts, key=lambda d: counts[d])
    if counts[best] == 0:
        raise DataError(
            f"{label} header line {header[:80]!r} has no ',', tab or ';' in it, so it names a "
            "single column. Fix: export a delimited file with one column per field."
        )
    return best


def _column_resolver(columns: Sequence[str]) -> Resolver:
    lookup = {c.strip().casefold(): c for c in columns}

    def resolve(wanted: str | None) -> str | None:
        if wanted is None:
            return None
        return lookup.get(wanted.strip().casefold())

    return resolve


def _missing_column(label: str, wanted: str, role: str, columns: Sequence[str]) -> str:
    return (
        f"{label} has no {wanted!r} column for the {role}. Its columns are: "
        f"{', '.join(columns)}. Fix: rename the column, or pass "
        f"FieldMapping({role}='<your column>')."
    )


def _require_column(
    resolve: Resolver, wanted: str, role: str, label: str, columns: Sequence[str]
) -> str:
    found = resolve(wanted)
    if found is None:
        raise DataError(_missing_column(label, wanted, role, columns))
    return found


def _detect_option_shape(
    columns: Sequence[str], mapping: FieldMapping, resolve: Resolver, label: str
) -> tuple[str, list[str], list[tuple[str, str | None]]]:
    """Decide between the long and the wide option shape, or refuse to guess."""
    long_col = resolve(mapping.options)
    long_ids_col = resolve(mapping.option_ids)

    by_index: dict[int, str] = {}
    id_by_index: dict[int, str] = {}
    for column in columns:
        if column == long_col:
            continue
        id_match = _WIDE_OPTION_ID.match(column)
        if id_match:
            id_by_index[int(id_match.group(1))] = column
            continue
        match = _WIDE_OPTION.match(column)
        if match:
            by_index[int(match.group(1))] = column

    has_long = long_col is not None
    has_wide = len(by_index) >= 2

    if has_long and has_wide:
        raise DataError(
            f"{label} has both a {long_col!r} column and the wide columns "
            f"{', '.join(by_index[i] for i in sorted(by_index))}, so the option shape is "
            f"ambiguous. {_SHAPES_HELP} Fix: keep one shape, or name the one you mean with "
            "FieldMapping(options='<column>')."
        )
    if has_long:
        cols = [long_col] if long_ids_col is None else [long_col, long_ids_col]
        return "long", [c for c in cols if c is not None], []
    if has_wide:
        indices = sorted(by_index)
        expected = list(range(1, len(indices) + 1))
        if indices != expected:
            missing = sorted(set(expected) - set(indices))
            raise DataError(
                f"{label} has wide option columns numbered {indices} with a gap: "
                f"option_{missing[0]} is missing. Fix: number them 1..n with no gaps, or move "
                f"to the long shape. {_SHAPES_HELP}"
            )
        return "wide", [], [(by_index[i], id_by_index.get(i)) for i in indices]
    if len(by_index) == 1:
        only = by_index[next(iter(by_index))]
        raise DataError(
            f"{label} has {only!r} but no second option column, and a criterion needs at least "
            f"two options. {_SHAPES_HELP} Fix: add 'option_2', or use the long shape."
        )
    raise DataError(
        f"{label} has no options column. Its columns are: {', '.join(columns)}. "
        f"{_SHAPES_HELP} Fix: add one of those shapes, or point at your own column with "
        "FieldMapping(options='<column>')."
    )


def _choose_option_delimiter(scan: _CsvScan, options_col: str, label: str) -> str:
    """Pick the delimiter that splits *every* options cell into two or more parts."""
    candidates = ", ".join(repr(d) for d in OPTION_DELIMITERS)
    if not scan.filled:
        raise DataError(
            f"{label}: every cell of column {options_col!r} is empty. Fix: list each row's "
            f"options there, separated by one of {candidates}."
        )
    if scan.viable:
        LOGGER.debug("option delimiter for %s detected as %r", options_col, scan.viable[0])
        return scan.viable[0]
    if scan.unsplittable is not None:
        line, cell = scan.unsplittable
        raise DataError(
            f"{label} row {line}: column {options_col!r} holds {cell[:60]!r}, which none of "
            f"the delimiters {candidates} splits into two or more options. Fix: separate the "
            "options with '|', or pass FieldMapping(option_delimiter='<your separator>')."
        )
    shown = " and ".join(f"row {line} holds {cell[:40]!r}" for line, cell in scan.examples)
    raise DataError(
        f"{label}: column {options_col!r} does not use one delimiter throughout ({shown}), so "
        f"none of {candidates} splits every row. Fix: use the same separator on every row, or "
        "pass FieldMapping(option_delimiter='<your separator>')."
    )


def _options_from_long(
    row: Mapping[str, str], cols: Sequence[str], delimiter: str, where: str
) -> list[dict[str, str]]:
    options_col = cols[0]
    ids_col = cols[1] if len(cols) > 1 else None
    cell = (row.get(options_col) or "").strip()
    if not cell:
        raise DataError(
            f"{where}: column {options_col!r} is empty. Fix: list the options the model may "
            f"choose between, separated by {delimiter!r}."
        )
    descriptions = [part.strip() for part in cell.split(delimiter) if part.strip()]
    if len(descriptions) < 2:
        raise DataError(
            f"{where}: column {options_col!r} holds {cell[:60]!r}, which splits on "
            f"{delimiter!r} into {len(descriptions)} option(s). A criterion needs at least two. "
            "Fix: list every option the row permits, or drop the row."
        )

    ids: list[str] | None = None
    if ids_col:
        id_cell = (row.get(ids_col) or "").strip()
        if id_cell:
            ids = [part.strip() for part in id_cell.split(delimiter) if part.strip()]
            if len(ids) != len(descriptions):
                raise DataError(
                    f"{where}: column {ids_col!r} lists {len(ids)} id(s) but {options_col!r} "
                    f"lists {len(descriptions)} option(s) ({ids} vs {descriptions}). Fix: the "
                    f"two columns must line up one for one, both split on {delimiter!r}."
                )
    return _pair(descriptions, ids, where)


def _options_from_wide(
    row: Mapping[str, str], cols: Sequence[tuple[str, str | None]], where: str
) -> list[dict[str, str]]:
    descriptions: list[str] = []
    ids: list[str] = []
    seen_blank: str | None = None
    for column, id_column in cols:
        value = (row.get(column) or "").strip()
        if not value:
            seen_blank = seen_blank or column
            continue
        if seen_blank is not None:
            raise DataError(
                f"{where}: column {seen_blank!r} is empty but {column!r} holds {value[:40]!r}, "
                "so the options have a hole in the middle. Fix: fill the gap, or move the "
                "later options left so they run option_1..option_n without a break."
            )
        descriptions.append(value)
        ids.append((row.get(id_column) or "").strip() if id_column else "")
    if len(descriptions) < 2:
        filled = [c for c, _ in cols if (row.get(c) or "").strip()]
        raise DataError(
            f"{where}: only {len(descriptions)} of the option columns "
            f"{', '.join(c for c, _ in cols)} is filled in ({filled or 'none'}). A criterion "
            "needs at least two options. Fix: fill the remaining option columns, or drop the row."
        )
    return _pair(descriptions, ids if any(ids) else None, where)


def _pair(
    descriptions: Sequence[str], ids: Sequence[str] | None, where: str
) -> list[dict[str, str]]:
    """Attach an id to each description, inventing stable ones where none were given."""
    out: list[dict[str, str]] = []
    used: set[str] = set()
    for index, description in enumerate(descriptions):
        given = (ids[index].strip() if ids and index < len(ids) else "") or ""
        candidate = _slug(given) if given else _slug(description)
        if not candidate:
            candidate = f"option_{index + 1}"
        if candidate in used:
            if given:
                raise DataError(
                    f"{where}: option id {given!r} appears twice. Fix: give every option in a "
                    "row a distinct id."
                )
            candidate = _distinct_id(candidate, index + 1, used)
        used.add(candidate)
        out.append({"id": candidate, "description": description})
    return out


def _distinct_id(base: str, number: int, used: set[str]) -> str:
    """`base` with a numeric suffix that makes it unique and keeps it within 64 chars.

    The suffix is fitted *inside* the limit. Appending and then truncating would cut the
    suffix straight back off a 64-character slug -- two long option texts sharing their
    first 64 characters is ordinary -- and hand back the very id that collided.
    """
    while True:
        suffix = f"_{number}"
        candidate = base[: 64 - len(suffix)].rstrip("_") + suffix
        if candidate not in used:
            return candidate
        number += 1


@dataclass
class _NumericAnswer:
    position: int  # index into the list of decisions being built
    where: str
    value: int
    ids: tuple[str, ...]


@dataclass
class _IndexedAnswers:
    """Settles whether a source's numeric answers are 0- or 1-based -- once per source.

    Deciding row by row is how labels get silently corrupted: in a 1-based column over
    three options, "3" can only be 1-based and lands on the third option, while "1" and
    "2" are also valid 0-based positions and land one option too far along. The column
    has one convention, so the evidence is pooled: a 0 proves 0-based, a value equal to
    the row's option count proves 1-based, both at once is an error, and neither is
    read as 0-based with a warning that says so.
    """

    rows: list[_NumericAnswer] = field(default_factory=list)

    def note(self, position: int, where: str, value: int, ids: Sequence[str]) -> None:
        self.rows.append(_NumericAnswer(position, where, value, tuple(ids)))

    def settle(self, decisions: list[Decision], label: str) -> None:
        """Rewrite, in place, the answers of numeric rows if the source is 1-based."""
        if not self.rows:
            return
        one_based = next((r for r in self.rows if r.value == len(r.ids)), None)
        zero_based = next((r for r in self.rows if r.value == 0), None)
        if one_based is not None and zero_based is not None:
            raise DataError(
                f"{label} mixes 0-based and 1-based answer indices: {zero_based.where} holds "
                f"0, which only a 0-based index can be, and {one_based.where} holds "
                f"{one_based.value} for a row with {len(one_based.ids)} options, which only a "
                "1-based index can be. Fix: use one convention throughout, or give the "
                "answer as an option id or letter."
            )
        if one_based is not None:
            for row in self.rows:
                decisions[row.position] = replace(
                    decisions[row.position], answer_id=row.ids[row.value - 1]
                )
            LOGGER.info(
                "%s: %d numeric answer(s) read as 1-based, because %s holds %d for %d options",
                label, len(self.rows), one_based.where, one_based.value, len(one_based.ids),
            )
        elif zero_based is None:
            LOGGER.warning(
                "%s: %d numeric answer(s) could be 0-based or 1-based -- no row holds 0 and "
                "none holds its own option count. They were read as 0-based. Give the answer "
                "as an option id or letter to remove the doubt.",
                label, len(self.rows),
            )


def _clip(text: str, limit: int = 60) -> str:
    """At most `limit` characters of a cell, for quoting in an error message.

    A mis-mapped column can point the answer at the evidence itself, and an error
    message ends up in logs and tickets; it should identify the cell, not reproduce it.
    """
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _whole_number(value: str) -> int | None:
    """``"2"`` and ``"2.0"`` (a float column is how pandas writes integers beside a
    blank) as an int; anything else, including non-ASCII digits, as None."""
    match = _WHOLE_NUMBER.match(value)
    return int(match.group(1)) if match else None


def _resolve_answer(
    raw: Any,
    options: Sequence[Mapping[str, str]],
    where: str,
    column: str,
    *,
    numeric: _IndexedAnswers | None = None,
    position: int = 0,
    positional: bool = False,
) -> str | None:
    """Accept whatever a ticketing system calls the gold: id, letter, index or text.

    ``positional`` marks a field that is a 0-based position by definition
    (``answer_index``, ``label``), which is never reinterpreted as 1-based.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    value = str(raw).strip()
    if not value:
        return None

    # A blank-word ("unknown", "n/a", "none", ...) means "unlabelled" only if it is not
    # one of this row's own answers. Checked first, it silently unlabelled every row
    # whose gold really is an option called `unknown` -- so matching gets the first say.
    ids = [o["id"] for o in options]
    if value in ids:
        return value
    folded = value.casefold()
    for option in options:
        if option["id"].casefold() == folded:
            return option["id"]
    if len(value) == 1 and value.isalpha():
        index = ord(value.upper()) - ord("A")
        if 0 <= index < len(options):
            return ids[index]
    number = _whole_number(value)
    if number is not None and positional:
        if 0 <= number < len(options):
            return ids[number]
        raise DataError(
            f"{where}: field {column!r} holds {number}, but it is a 0-based position and this "
            f"row has {len(options)} options (valid positions are 0..{len(options) - 1}). "
            "Fix: correct the position, or give the answer as an option id."
        )
    if number is not None and 0 <= number <= len(options):
        if numeric is None:
            return ids[number] if number < len(options) else ids[number - 1]
        # Provisionally 0-based; `_IndexedAnswers.settle` rewrites the whole source as
        # 1-based if any row proves that is what the column means.
        numeric.note(position, where, number, ids)
        return ids[min(number, len(options) - 1)]
    for option in options:
        if option["description"].strip().casefold() == folded:
            return option["id"]
    slugged = _slug(value)
    if slugged and slugged in ids:
        return slugged
    if folded in _BLANK_WORDS:
        return None

    raise DataError(
        f"{where}: column {column!r} holds {_clip(value)!r}, which is not one of this row's "
        "options. "
        f"It permits the ids {', '.join(ids)} (descriptions: "
        f"{', '.join(repr(o['description'][:24]) for o in options)}). Fix: use an option id, its "
        f"answer letter A-{chr(ord('A') + len(options) - 1)}, its 0-based index, or the exact "
        "option text; leave the cell blank for an unlabelled row."
    )


# ------------------------------------------------------------------- shared row logic


def _decision_from_mapping(
    raw: Mapping[str, Any],
    mapping: FieldMapping,
    label: str,
    line_no: int,
    numeric: _IndexedAnswers | None = None,
    *,
    position: int = 0,
) -> Decision:
    """Turn one JSON/Parquet/HF row into a Decision, honouring the field mapping."""
    row = dict(raw)
    row_id = _first(row, [mapping.id] if mapping.id else ["id"])
    where = _where(label, line_no, str(row_id) if row_id else "")

    if mapping.id and mapping.id not in row:
        raise DataError(
            f"{where}: no {mapping.id!r} field, but FieldMapping(id={mapping.id!r}) asked for "
            f"one. Its fields are: {', '.join(sorted(map(str, row)))}. Fix: drop the mapping, "
            "or add the field."
        )

    state = _first(row, [mapping.state, "state", "evidence", "text"])
    if state is None or not str(state).strip():
        raise DataError(
            f"{where}: no non-empty {mapping.state!r} field. Its fields are: "
            f"{', '.join(sorted(map(str, row)))}. Fix: rename the column, or pass "
            f"FieldMapping(state='<your field>')."
        )
    question = _first(row, [mapping.question, "question", "criterion_question"])
    if question is None or not str(question).strip():
        raise DataError(
            f"{where}: no non-empty {mapping.question!r} field. Its fields are: "
            f"{', '.join(sorted(map(str, row)))}. Fix: rename the column, or pass "
            "FieldMapping(question='<your field>')."
        )

    raw_options = _first(row, [mapping.options, "options", "choices"])
    options = _normalise_options(raw_options, where, mapping)

    payload: dict[str, Any] = {
        "state": str(state),
        "question": str(question),
        "options": options,
        "key": _criterion_key(_first(row, [mapping.key, "key", "criterion"]), str(question)),
    }
    if row_id is not None and str(row_id).strip():
        payload["id"] = str(row_id)
    group = _first(row, [mapping.group] if mapping.group else ["group_id", "group"])
    if group is not None and str(group).strip():
        payload["group_id"] = str(group)

    answer_keys = [mapping.answer, "answer_id", "answer", "gold"]
    answer = _first(row, answer_keys)
    if answer is not None:
        column = next((k for k in answer_keys if k and row.get(k) is not None), mapping.answer)
        payload["answer_id"] = _resolve_answer(
            answer, options, where, column, numeric=numeric, position=position
        )
    else:
        # `answer_index` and `label` are 0-based positions by definition (see
        # schema.decision_from_dict), so they are never second-guessed as 1-based.
        for positional_key in ("answer_index", "label"):
            if row.get(positional_key) is not None:
                payload["answer_id"] = _resolve_answer(
                    row[positional_key], options, where, positional_key, positional=True
                )
                break

    for key, value in row.items():
        if isinstance(key, str) and key.startswith("meta_"):
            payload[key] = value
    return _build(payload, where)


def _first(row: Mapping[str, Any], names: Sequence[str | None]) -> Any:
    for name in names:
        if name is not None and name in row and row[name] is not None:
            return row[name]
    return None


def _normalise_options(raw: Any, where: str, mapping: FieldMapping) -> list[dict[str, str]]:
    if raw is None:
        raise DataError(
            f"{where}: no {mapping.options!r} field. Fix: give the closed set of answers, either "
            "as a list of {{'id': ..., 'description': ...}} objects or as a list of strings."
        )
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as error:
                raise DataError(
                    f"{where}: field {mapping.options!r} looks like JSON but does not parse "
                    f"({error.msg}). Fix: quote it properly, or use a plain delimited list."
                ) from error
        else:
            delimiter = mapping.option_delimiter or next(
                (d for d in OPTION_DELIMITERS if d in text), "|"
            )
            raw = [part.strip() for part in text.split(delimiter) if part.strip()]

    if not isinstance(raw, (list, tuple)):
        raise DataError(
            f"{where}: field {mapping.options!r} holds a {type(raw).__name__}, not a list. "
            "Fix: give a list of options, one entry per permitted answer."
        )
    if len(raw) < 2:
        raise DataError(
            f"{where}: field {mapping.options!r} lists {len(raw)} option(s). A criterion needs "
            "at least two. Fix: list every option the row permits, or drop the row."
        )

    descriptions: list[str] = []
    ids: list[str] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, Mapping):
            description = str(entry.get("description") or entry.get("text") or "").strip()
            if not description:
                raise DataError(
                    f"{where}: option {index + 1} has no 'description' for the model to read "
                    f"(got {dict(entry)!r}). Fix: every option needs description text."
                )
            descriptions.append(description)
            ids.append(str(entry.get("id") or "").strip())
        else:
            descriptions.append(str(entry).strip())
            ids.append("")
    return _pair(descriptions, ids if any(ids) else None, where)


def _criterion_key(raw: Any, question: str) -> str:
    if raw is not None and str(raw).strip():
        slug = _slug(str(raw))
        if slug:
            return slug
    return _slug(question) or "decision"


def _slug(text: str) -> str:
    slug = _SLUG_STRIP.sub("_", text.strip().lower()).strip("_")
    return slug[:64].strip("_")


def _where(label: str, line_no: int, row_id: str) -> str:
    ident = f", id {row_id!r}" if row_id else ""
    return f"{label} row {line_no}{ident}"


def _build(payload: dict[str, Any], where: str) -> Decision:
    try:
        return decision_from_dict(payload)
    except SchemaError as error:
        _reraise(where, error)


def _reraise(where: str, error: SchemaError) -> NoReturn:
    raise DataError(f"{where}: {error}") from error
