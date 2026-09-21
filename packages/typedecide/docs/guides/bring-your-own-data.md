# Bring your own data

Point at your columns; do not rename them first. JSONL, JSON, CSV/TSV, Parquet and
Hugging Face dataset ids all load into the same [`Decision`](../reference/api/schema.md)
type through one function.

```python
from typedecide import load_decisions
from typedecide.data import FieldMapping

decisions = load_decisions(
    "export.csv",
    mapping=FieldMapping(
        id="Ticket ID", state="Ticket Body", question="Decision",
        options="Choices", answer="Chosen", group="Case",
    ),
)
```

The same from the command line, for any subcommand that reads data:

```bash
typedecide validate export.csv \
  --col-id "Ticket ID" --col-state "Ticket Body" --col-question "Decision" \
  --col-options "Choices" --col-answer "Chosen" --col-group "Case"
```

Loading is strict. Every failure raises [`DataError`](../reference/api/errors.md)
naming the file, the row, the column, the offending value and a fix.

## Formats

| Format | Suffixes | `format=` | Notes |
|---|---|---|---|
| JSON Lines | `.jsonl`, `.ndjson` | `jsonl` | One object per line. Blank lines are skipped. Streamed |
| JSON | `.json` | `json` | An array of objects, or an object holding the array under `decisions`, `data`, `rows` or `examples`. Read whole; use JSONL for anything large |
| CSV / TSV | `.csv`, `.tsv` | `csv` | Header required. The field delimiter is sniffed from the header (`,`, tab or `;`); `.tsv` forces tab |
| Parquet | `.parquet`, `.pq` | `parquet` | Needs `pip install "typedecide[parquet]"` |
| Hugging Face dataset | `owner/name` or `owner/name:split` | `hf` | Needs `pip install "typedecide[train]"`. The split defaults to `train` |

The format is inferred from the suffix. A string that is not an existing path and
looks like `owner/name` is treated as a Hugging Face dataset id. For a file with an
unrecognised suffix, or a dataset id with no `/` in it, pass `format=` (`--format`)
explicitly. Files are read as UTF-8, and a byte-order mark is tolerated.

!!! note "Hugging Face datasets and remote code"
    `typedecide` never enables `trust_remote_code`. A dataset that needs a loading
    script is refused by `datasets`; export it to JSONL or Parquet first.

## `FieldMapping`

| Field | Default | Meaning |
|---|---|---|
| `state` | `"state"` | The evidence text |
| `question` | `"question"` | The question asked about it |
| `options` | `"options"` | The closed set of answers |
| `answer` | `"answer_id"` | The gold answer |
| `group` | `None` | The group id. `None` means use the conventional column if the file has one |
| `id` | `None` | The row id. `None` means use `id` if present, otherwise hash the content |
| `option_ids` | `"option_ids"` | CSV long shape only: ids parallel to `options` |
| `key` | `"key"` | The criterion key. Derived by slugging the question when absent |
| `option_delimiter` | `None` | Separator inside `options` and `option_ids`. `None` auto-detects |

The CLI exposes the first six as `--col-*`. The last three are Python-only.

The loader also recognises common alternative names, so a mapping is often
unnecessary:

| Role | JSON / Parquet / Hugging Face rows | CSV columns (case-insensitive) |
|---|---|---|
| state | mapped name, `state`, `evidence`, `text` | mapped name (required) |
| question | mapped name, `question`, `criterion_question` | mapped name (required) |
| options | mapped name, `options`, `choices` | mapped name, or `option_1`, `option_2`, ... |
| answer | mapped name, `answer_id`, `answer`, `gold`; otherwise `answer_index` or `label` | mapped name, `answer`, `label` |
| key | mapped name, `key`, `criterion` | mapped name, `criterion` |
| group | mapped name, or `group_id`, `group` | mapped name, or `group_id` |
| id | mapped name, or `id` | mapped name, or `id` |

If you name an `id` or `group` column explicitly and it does not exist, that is an
error. If you leave them unset and the conventional column is absent, the field is
left unset.

## CSV: two ways to express options

A flat table has no natural way to hold a list, so the loader accepts two shapes and
refuses to guess if it sees both.

=== "Long: one delimited column"

    ```csv
    id,state,question,options,option_ids,answer_id,group_id
    T-1,"I was charged twice.",Which queue?,Billing|Shipping|Accounts,bill|ship|acct,bill,C-1
    T-2,"My parcel is lost.",Which queue?,Billing|Shipping|Accounts,bill|ship|acct,B,C-2
    ```

    - `options` holds the option **descriptions**, delimited.
    - `option_ids` is optional. When present it must line up with `options` one for
      one in every row.
    - The delimiter is auto-detected as the first of `||`, `|`, `;`, tab, `,` that
      splits **every** options cell into two or more parts. Set
      `FieldMapping(option_delimiter=...)` when option text contains one of those
      characters or you use another separator.

=== "Wide: one column per option"

    ```csv
    id,state,question,option_1,option_2,option_3,option_1_id,option_2_id,option_3_id,answer
    w1,"I was charged twice.",Which queue?,Billing,Shipping,Accounts,bill,ship,acct,ship
    w2,"Is this urgent?",Urgent?,Yes,No,,yes,no,,A
    ```

    - Columns named `option_1`, `option_2`, ... (also `option 1`, `option-1`,
      `Option1`), numbered from 1 with no gaps.
    - `option_N_id` columns are optional.
    - A row may use fewer options than there are columns by leaving **trailing** cells
      blank. A blank in the middle is an error.

When no ids are given, they are derived by slugging each description:
`"Billing and payments"` becomes `billing_and_payments`. If two descriptions slug to
the same id, the later one gets a numeric suffix. Explicit duplicate ids are an error.

Other CSV rules: fully blank lines are skipped; a row with more fields than the header
is an error that shows the start of the extra field (usually an unquoted comma); a
short row is padded with blanks; duplicate header names are an error.

!!! note "`meta_` columns are not carried through from CSV"
    JSON, JSONL, Parquet and Hugging Face rows keep any field whose name starts with
    `meta_` in `Decision.meta`. The CSV loader reads only the columns it maps.

## JSON, JSONL, Parquet and Hugging Face rows

`options` may be any of:

```json
{"options": [{"id": "bill", "description": "Billing"}, {"id": "ship", "description": "Shipping"}]}
{"options": [{"id": "bill", "text": "Billing"}, {"id": "ship", "text": "Shipping"}]}
{"options": ["Billing", "Shipping"]}
{"options": "[\"Billing\", \"Shipping\"]"}
{"options": "Billing|Shipping"}
```

That is: objects with `id` and `description` (or `text`), plain strings, a
JSON-encoded string, or a delimited string. Ids are derived from descriptions when
absent.

## Answer formats accepted

The gold may be written in whichever form your system exports. Forms are tried in
this order:

1. An exact **option id**, then a case-insensitive id match.
2. A single **answer letter**, `A` for the first option, case-insensitive.
3. A whole **number**, as an index (`2`, and also `2.0`, which is how pandas writes an
   integer column that has blanks).
4. The exact **option text**, case-insensitive.
5. Text that **slugs to an option id** (`"Billing and Payments"` matches
   `billing_and_payments`).

Anything else is a `DataError` listing the permitted ids and descriptions.

!!! warning "Blank-like words mean unlabelled"
    An empty cell, and the words `na`, `n/a`, `nan`, `none`, `null`, `-`, `?` and
    `unknown` (any case), are read as **no answer**. This check happens first. If one
    of your options has the id or text `unknown` or `none`, a gold cell containing
    that word is read as unlabelled, not as that option. Use the answer letter for
    those rows, or choose a different id such as `cannot_tell`.

!!! warning "Numeric answers: 0-based or 1-based is decided once per file"
    A number is ambiguous: in a three-option row, `1` could mean the first option
    (1-based) or the second (0-based). The loader pools the evidence across the whole
    source:

    - a row holding `0` proves the column is 0-based;
    - a row holding its own option count (`3` of 3) proves it is 1-based, and **every**
      numeric answer in the source is then read as 1-based;
    - both in one source is a `DataError`;
    - neither means the answers are read as **0-based**, and a warning is logged
      saying so.

    The fields `answer_index` and `label` in JSON-like rows are 0-based by definition
    and are never reinterpreted. If your export is 1-based and you cannot be sure a
    row ever reaches the last option, convert the column to option ids or letters.
    After loading, compare the `label balance` block of `typedecide validate` with
    what you expect.

## Criterion keys

Rows with the same `key` are treated as the same criterion: they share a label space
in `validate`, in the split's class check and in `per_criterion` results. If your data
has no key column, the key is the slugged question, truncated to 64 characters, so two
wordings of one question become two criteria. Supply a key column when wording varies.

## Writing decisions back out

```python
from pathlib import Path

from typedecide import load_decisions
from typedecide.data import write_decisions

decisions = load_decisions("tickets.csv")
write_decisions(decisions, Path("data/all.jsonl"))    # JSONL
write_decisions(decisions, Path("data/all.json"))     # a JSON array
```

A path ending in `.json` gets an array; anything else gets JSONL. Parent directories
are created. The file is written beside the target and renamed over it, so an
interrupted write does not leave a truncated dataset. Converting a customer export to
JSONL once, and using that from then on, removes every ambiguity described above.

## A checklist for a new export

1. `typedecide validate export.csv [--col-* flags]` loads without an `error:` line.
2. `labelled` in the counts matches your expectation. A shortfall means some answers
   were read as blank.
3. The `label balance` keys are the criteria you expect, and the option ids under each
   are the ones you expect.
4. `groups` is smaller than `decisions` if several rows share a state. If it equals
   `decisions` and rows do share states, you have not mapped the group column.
5. Convert to JSONL with `typedecide split` or `write_decisions` and keep that file.

If step 1 fails, use the
[validation runbook](../runbooks/validation-fails.md).
