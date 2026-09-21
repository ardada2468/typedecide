# Handle errors

Every failure the library anticipates raises one of its own exceptions, and the
message names the thing that is wrong and what would fix it. Anything else (an
out-of-memory error from torch, a network error from the Hub) propagates unchanged.

## The hierarchy

```text
Exception
└── TypeDecideError                  catch this to catch anything raised deliberately
    ├── SchemaError                  a decision, criterion or option is not well formed
    ├── DataError                    a dataset could not be read, or is unusable as given
    ├── PromptError                  the prompt and the tokenizer disagree
    ├── TrainingError                training cannot start, or cannot be trusted
    ├── ExportError                  a model could not be turned into the artefact asked for
    ├── ConfigError                  a config file or invocation does not describe a runnable job
    └── EvaluationError              scoring started from usable inputs and could not produce a score
        └── ScorerError              a ScoreFn broke its contract (also a ConfigError; see below)
```

All of them are importable from the top-level package and from `typedecide.errors`:

```python
from typedecide import (
    ConfigError, DataError, EvaluationError, ExportError, PromptError,
    SchemaError, ScorerError, TrainingError, TypeDecideError,
)
```

## Which class, from where

| Exception | Raised by | Typical causes |
|---|---|---|
| `SchemaError` | `typedecide.schema` | Invalid option id or criterion key; fewer than 2 or more than 20 options; duplicate option id; `answer_id` that is not one of the options; empty `id`, `state` or `question`; a `reordered()` argument that is not a permutation |
| `DataError` | `typedecide.data` (loaders, `group_split`, `all_permutations`, `Finding`), `typedecide.evaluation.metrics`, `evaluate_with_scorer` | Unreadable or malformed file; missing column; ambiguous option shape; answer that matches no option; a missing `parquet` or `datasets` dependency; a split that would hide a class; an eval set with no labelled rows; mismatched metric inputs |
| `PromptError` | `typedecide.prompt` | No stable answer-letter continuation for this tokenizer; a prompt that encodes to zero tokens; a state prefix too short to reuse |
| `TrainingError` | `typedecide.training` | Missing `torch`, `transformers` or `peft`; unlabelled training rows; an empty dataset; a criterion too long for `max_length`; LoRA cannot attach; tokenizer or model cannot load. A `PromptError` during training is re-raised as `TrainingError` with the decision id attached |
| `ExportError` | `typedecide.export` | Missing exporter dependency; adapter directory not found; adapter and base mismatch; opset below 14; invalid `keep` list; a failed untie check; Optimum export failure; invalid or mismatched `slot_map.json`; unknown quantisation mode |
| `ConfigError` | `TrainConfig`, `typedecide.evaluation` | Unknown or missing config key; out-of-range value; unreadable config file; unknown `debias` mode; `permutation` with more than 6 options; `batch_size < 1`; a missing `torch`/`transformers`/`peft` when scoring; a tokenizer with neither pad nor EOS token |
| `ScorerError` | `evaluate_with_scorer`, and therefore `evaluate` | A `ScoreFn` returned the wrong number of vectors, a vector of the wrong length, a negative value, a non-finite value (`NaN` or `inf`, usually a half-precision overflow in the model), or something that is not a sequence of numbers |

!!! note "Loader errors are always `DataError`"
    When a row fails schema validation during loading, the `SchemaError` is wrapped in
    a `DataError` that adds the file and row (`export.csv row 12, id 'T-9': ...`), with
    the original as `__cause__`. You only see a bare `SchemaError` when you construct
    `Option`, `Criterion` or `Decision` yourself or call `decision_from_dict`.

!!! note "`ScorerError` and backward compatibility"
    `ScorerError` inherits from both `EvaluationError` and `ConfigError`. Version 0.1.0
    raised `ConfigError` for a misbehaving scorer, so `except ConfigError` written
    against it keeps working. That base is deprecated. Catch `EvaluationError` in new
    code.

## Patterns

### One boundary for everything deliberate

```python
import sys

from typedecide import TypeDecideError, group_split, load_decisions, validate


def prepare(path: str):
    decisions = load_decisions(path)
    report = validate(decisions)
    if not report.ok:
        raise SystemExit(report.render())
    return group_split(decisions, eval_fraction=0.2, seed=0)


try:
    train, held = prepare("tickets.csv")
except TypeDecideError as error:
    # The message is the diagnosis. A traceback would only bury it.
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)
```

This is what the CLI does.

### Reacting to specific classes

```python
from typedecide import DataError, group_split, load_decisions

decisions = load_decisions("tickets.jsonl")

for seed in range(20):
    try:
        train, held = group_split(decisions, eval_fraction=0.2, seed=seed)
        break
    except DataError:
        continue
else:
    raise SystemExit("no seed covers every class; see the split-refuses runbook")
```

```python
from pathlib import Path

from typedecide import ConfigError
from typedecide.training import TrainConfig

try:
    config = TrainConfig.from_file(Path("train.yaml"))
except ConfigError as error:
    raise SystemExit(f"bad training config: {error}")
```

### A validation report is not an exception

`validate` never raises for a bad dataset. It returns a
[`DataReport`](../reference/api/data/validate.md), and you decide what to do with
`report.ok`. Only *loading* raises.

### Match on classes and codes, never on message text

Message wording is not part of the public API and improves between releases. Branch on
the exception class, on `Finding.code`, or on the CLI exit status.

## What the CLI does

| Situation | Output | Exit status |
|---|---|---:|
| Success | The function's result, on standard output | `0` |
| `validate` finds `error` findings | The report | `1` |
| Any `TypeDecideError` | `error: <message>` on standard error, no traceback | `2` |
| An `OSError`, such as an unwritable `--out` or `--json-out` | `error: <message>` on standard error, no traceback | `2` |
| ++ctrl+c++ | `interrupted` | `130` |
| Any other exception | A Python traceback | `1` |

Run with `-v` before the subcommand (`typedecide -v train ...`) for debug logging.

## Logging

The library logs under the `typedecide` namespace
(`typedecide.data.loaders`, `typedecide.training.trainer`,
`typedecide.evaluation.runner`, and so on) and never prints outside the CLI. It does
not configure logging for you; it installs a `NullHandler` so nothing is emitted until
your application adds a handler.

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("typedecide").setLevel(logging.DEBUG)
```

Some conditions are deliberately warnings, not errors, because the run can still be
trusted: unlabelled rows skipped by evaluation, a state truncated from the left,
numeric answers of ambiguous base, a slot map whose separator could not be reproduced.
Read the log of a long run.

## Missing optional dependencies

A missing extra is reported as one of the classes above, never as a bare
`ImportError`, and the message contains the `pip install` command. The mapping is in
[Install](../getting-started/install.md#what-happens-when-an-extra-is-missing).
