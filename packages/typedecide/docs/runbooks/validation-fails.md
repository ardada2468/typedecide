# Validation fails on a customer export

## Symptoms

One of:

- `typedecide validate export.csv` prints `error: ...` and exits with status **2**.
  The file could not be loaded.
- It prints `Dataset report: NOT USABLE AS GIVEN` and exits with status **1**. The file
  loaded, and the report contains `error` findings.

## Diagnosis

The exit status tells you which half of the problem you have.

```bash
typedecide validate export.csv --json-out report.json; echo "exit=$?"
```

**Exit 2: a load error.** The message names the file, the row, the column, the
offending value and a fix. Match its opening against this table:

| Message contains | Cause |
|---|---|
| `has no 'state' column for the state. Its columns are: ...` | Your columns have other names |
| `has no options column` | No `options` column and no `option_1`, `option_2`, ... columns |
| `has both a ... column and the wide columns` | Both option shapes are present |
| `wide option columns numbered [...] with a gap` | `option_1`, `option_3` with no `option_2` |
| `which none of the delimiters ... splits into two or more options` | One options cell uses a separator the others do not |
| `lists N id(s) but ... lists M option(s)` | `option_ids` and `options` do not line up in that row |
| `has N fields but the header names M` | An unquoted delimiter inside a field, usually a comma in the ticket text |
| `holds '...', which is not one of this row's options` | The gold answer does not match any option |
| `header repeats the column name(s)` | Duplicate header |
| `Cannot infer a format from` | Unrecognised file suffix |
| `is not valid JSON` | Wrong `--format`, or a broken line |

**Exit 1: report errors.** List them:

```bash
python - <<'PY'
import json
for f in json.load(open("report.json"))["findings"]:
    if f["severity"] == "error":
        print(f["code"], len(f["decisions"]), "rows:", f["decisions"][:5])
PY
```

## Fix

### Load errors

- **Different column names.** Point at them; do not rename the file:

    ```bash
    typedecide validate export.csv \
      --col-id "Ticket ID" --col-state "Ticket Body" --col-question "Decision" \
      --col-options "Choices" --col-answer "Chosen" --col-group "Case"
    ```

    Column matching is case-insensitive. Quote names containing spaces.

- **Options in an unusual separator.** The auto-detected delimiter must split *every*
  options cell into two or more parts. Candidates are tried in this order: `||`, `|`,
  `;`, tab, `,`. If your separator is something else, or option text itself contains
  one of those characters, set it explicitly from Python:

    ```python
    from typedecide import load_decisions
    from typedecide.data import FieldMapping

    mapping = FieldMapping(
        state="Ticket Body", question="Decision", options="Choices", answer="Chosen",
        option_delimiter=" / ",          # options written as "Billing / Shipping / Accounts"
    )
    decisions = load_decisions("export.csv", mapping=mapping)
    ```

- **Too many fields in a row.** Re-export with proper CSV quoting, or as TSV. The
  message shows the start of the extra field, which identifies the unquoted text.

- **Answer is not an option.** Accepted forms are the option id, an answer letter, a
  0-based index, or the exact option text. Common causes are trailing qualifiers
  (`Billing (escalated)`), a 1-based index column, and a label from an older option
  set. See the warning about indices in
  [Bring your own data](../guides/bring-your-own-data.md#answer-formats-accepted).

- **Wrong format guess.** Pass `--format csv` (or `jsonl`, `json`, `parquet`, `hf`).

### Report errors

| Code | Fix |
|---|---|
| `duplicate_id` | Make the id column unique, or stop mapping it (`--col-id`) so the loader hashes the content |
| `no_labels` | The gold is in a column the loader did not look at. Pass `--col-answer` |
| `option_set_mismatch` | Rows sharing a criterion key list different option ids. Give the variants distinct keys or make the sets identical. If the customer's option list changed over time, split by period |
| `contradictory_labels` | Send the listed row ids back for adjudication. If the evidence really differs, the distinguishing text is missing from the state |
| `single_group` | The group column holds one value, often a tenant or batch id. Map `--col-group` to the ticket or case id instead |
| `empty` | The file has a header and nothing else, or a filter removed everything |

Warnings do not fail validation. Read them anyway:
[every code is explained here](../guides/validation-report.md#every-finding-code).

!!! warning "Do not fix contradictions by deleting one side at random"
    `contradictory_labels` means annotators disagreed about identical evidence. Which
    label survives decides what the model learns. Get a decision from whoever owns the
    policy.

## Verify

```bash
typedecide validate export.csv [your --col flags] --json-out report.json; echo "exit=$?"
```

- Exit status is `0` and the first line reads `Dataset report: OK`.
- `labelled` in the counts line matches the number of labelled rows you expect. A
  lower figure means some answers were read as blank.
- The `label balance` block shows the classes you expect, under the keys you expect.
- Then confirm the data survives the next step: `typedecide split export.csv --out data/ [same flags]`.
