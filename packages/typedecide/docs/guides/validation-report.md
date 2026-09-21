# Read a validation report

`validate` looks for the problems that make a training run a waste of an afternoon. It
needs no tokenizer and no model.

=== "CLI"

    ```bash
    typedecide validate tickets.csv --min-per-class 5 --json-out report.json
    ```

    Exit status: `0` if the report is OK, `1` if any finding has severity `error`,
    `2` if the file could not be loaded at all.

=== "Python"

    ```python
    from typedecide import load_decisions, validate

    report = validate(load_decisions("tickets.csv"), min_per_class=5,
                      max_state_tokens=1024)
    print(report.render())

    if not report.ok:
        for finding in report.by_severity("error"):
            print(finding.code, finding.decisions[:5])
    ```

## Anatomy of a report

```text
Dataset report: OK
  decisions=6, labelled=5, unlabelled=1, criteria=1, groups=6, unique_states=6, errors=0, warnings=2
  -- warnings (2) --
  [warning] rare_class: Criterion 'which_queue_should_handle_this_ticket' has class(es) below min_per_class=5: account_access=1, billing_and_payments=2, shipping_and_delivery=2. There are too few to both train on and hold out, so per-class scores will be noise. Fix: collect more, merge the class, or lower min_per_class deliberately.
  [warning] unlabelled: 1 of 6 rows have no answer and will be skipped by training and evaluation. Fix: label them, or filter them out deliberately.
      rows: T-6
  -- label balance --
    which_queue_should_handle_this_ticket: account_access=1 (20%), billing_and_payments=2 (40%), shipping_and_delivery=2 (40%)
```

- The first line is the verdict. `OK` means no `error` findings. It does **not** mean
  no warnings.
- `counts` summarises the dataset.
- Findings are grouped by severity, then sorted by code. Each lists up to 8 affected
  row ids and says how many more there are.
- `label balance` shows gold counts per criterion and option, including options with
  zero labels.

The three severities:

| Severity | Meaning | Effect on `report.ok` |
|---|---|---|
| `error` | The dataset is unusable as given | `False` |
| `warning` | It will run, and it will probably hurt you | none |
| `note` | Worth knowing | none |

## Every finding code

### Errors

`empty`
:   The dataset has no decisions. Check the source path and any filter you applied
    after loading.

`duplicate_id`
:   One or more decision ids appear more than once. Rows would overwrite each other in
    any id-keyed result, and a group split cannot be trusted because an unset
    `group_id` falls back to the id. Make the id column unique, or drop it and let the
    loader hash the content.

`no_labels`
:   No row has an answer, so there is nothing to train or score against. Often this
    means the gold is in a column the loader did not look at; pass
    `FieldMapping(answer="<your column>")` or `--col-answer`.

`option_set_mismatch`
:   Rows sharing a criterion `key` list different sets of option ids. The comparison
    ignores order, so this is about membership. A readout is defined over one closed
    set, so these are different criteria under one name. Give them distinct keys, or
    make the option set identical everywhere.

    !!! tip
        When `key` is absent it is derived by slugging the question, so two differently
        worded questions get different keys automatically. This error usually appears
        when you supply the key yourself.

`contradictory_labels`
:   Two or more labelled rows have the same criterion key and the same state (compared
    after collapsing whitespace and case-folding) but different answers. No model can
    be right on both, so they put a hard ceiling on accuracy that training cannot
    move. Reconcile the labels, or, if the evidence really differs, put the
    distinguishing part into the state text.

`single_group`
:   Every row shares one `group_id`, so a leakage-safe split has nowhere to cut. Group
    by the thing that must not be split (usually the state or source ticket), not by
    the dataset.

### Warnings

`unlabelled`
:   Some rows have no answer. Lists their ids.

    !!! warning "Training does not skip them"
        The message says unlabelled rows "will be skipped by training and evaluation".
        That is accurate for `evaluate`, which skips them and logs a warning, and for
        the eval set passed to `finetune`. It is **not** accurate for the training set:
        `finetune` raises `TrainingError` if any training row lacks an `answer_id`,
        on the principle that training silently on fewer rows than you passed is worse
        than stopping. `group_split` keeps unlabelled rows, so filter them before
        training:

        ```python
        from typedecide import group_split, load_decisions
        from typedecide.training import labelled_only

        train, held = group_split(load_decisions("tickets.csv"), seed=0)
        train, dropped_ids = labelled_only(train)
        ```

`question_mismatch`
:   Rows sharing a criterion key use different question wordings, so the prompt
    changes between rows that share a label space. Settle on one wording or split the
    key.

`imbalanced`
:   More than 70% of a criterion's labelled rows have the same answer. A model that
    always guesses that answer scores that share in plain accuracy while having learned
    only the prior. Collect or down-sample towards balance, and read
    `balanced_accuracy`, not `accuracy`.

`unused_option`
:   A criterion offers an option that no row is labelled with. The model is asked to
    consider an answer it never sees chosen, and no split can cover it. Label examples
    of it or remove it.

`rare_class`
:   A class has at least one and fewer than `min_per_class` labelled rows (default 5).
    There are too few to both train on and hold out, so per-class scores are noise.
    This is also the usual precursor to
    [the split refusing](../runbooks/split-refuses.md).

`long_state`
:   One or more states are estimated to exceed `max_state_tokens` (default 1024, the
    same as `TrainConfig.max_length`). The estimate is tokenizer-free and deliberately
    pessimistic: the larger of about 1.3 tokens per whitespace word and one token per
    four characters. Lists the rows, longest first.

    !!! note
        The message says it is the end of the ticket that is lost. Training in this
        library truncates the **start** of the state and protects the criterion and
        options; see
        [Answer-token-only loss](../concepts/answer-token-loss.md#truncation-comes-from-the-left-of-the-state).
        The CLI does not expose `max_state_tokens`; use the Python API to change it.

### Notes

`duplicate_rows`
:   Labelled rows repeat a state that is already present with the same answer.
    Duplicates inflate whichever split they land in, and leak across a split unless
    they share a `group_id`. De-duplicate, or give the copies a shared group.

## The JSON report

`--json-out` writes:

```json
{
  "counts": {"decisions": 6, "labelled": 5, "unlabelled": 1, "criteria": 1,
             "groups": 6, "unique_states": 6, "errors": 0, "warnings": 2},
  "label_balance": {"queue": {"billing": 2, "shipping": 2, "access": 1}},
  "findings": [
    {"severity": "warning", "code": "rare_class", "message": "...", "decisions": []}
  ],
  "ok": true
}
```

The finding `code`s are the stable part. Gate CI on codes and severities, not on
message text.

## Using it as a gate

```bash
typedecide validate tickets.csv --json-out report.json || {
  echo "dataset has errors; see report.json"; exit 1;
}
```

To also fail on particular warnings:

```python
from typedecide import load_decisions, validate

BLOCKING = {"imbalanced", "unlabelled"}

report = validate(load_decisions("tickets.csv"))
blocking = [f for f in report.findings if f.severity == "error" or f.code in BLOCKING]
if blocking:
    raise SystemExit("\n".join(f.render() for f in blocking))
```
