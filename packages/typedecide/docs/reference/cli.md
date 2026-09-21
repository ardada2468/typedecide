# CLI reference

The `typedecide` command is a thin wrapper over the library. Every subcommand maps to
one public function and prints what that function returned.

```text
typedecide [-h] [-v] {validate,split,train,evaluate,export,savings} ...
```

| Subcommand | Library function | Needs |
|---|---|---|
| [`validate`](#typedecide-validate) | `load_decisions`, `validate` | base install |
| [`split`](#typedecide-split) | `group_split`, `write_decisions` | base install |
| [`train`](#typedecide-train) | `TrainConfig.from_file`, `finetune` | `train` extra |
| [`evaluate`](#typedecide-evaluate) | `evaluate` | `train` extra |
| [`export`](#typedecide-export) | `answer_letter_token_ids`, `export_onnx` | `train` and `export` extras |
| [`savings`](#typedecide-savings) | `lm_head_savings` | base install |

You can also run it as a module: `python -m typedecide.cli ...`.

## Global options

| Flag | Meaning |
|---|---|
| `-h`, `--help` | Show help. Every subcommand also accepts `-h`. |
| `-v`, `--verbose` | Debug logging. The default level is `INFO`. |

!!! note "`-v` goes before the subcommand"
    It is an option of the top-level parser: `typedecide -v validate tickets.csv`.
    Placed after the subcommand, argparse rejects it.

Log lines are written to standard error in the format
`LEVEL logger.name: message`. Results are written to standard output.

## Exit status

| Code | Meaning |
|---:|---|
| `0` | Success. For `validate`: the report has no `error` findings. |
| `1` | `validate` only: the dataset loaded, and the report contains at least one `error` finding. |
| `2` | The library raised a `TypeDecideError`, or the operating system refused a read or write (for example an unwritable `--out` or `--json-out`). The message is printed to standard error as `error: <message>` with no traceback. Also argparse's own code for a usage error. |
| `130` | Interrupted with ++ctrl+c++. |

Any other exception is a bug or an environment failure and produces a normal Python
traceback.

## Input options

`validate`, `split`, `train` and `evaluate` read datasets and share these flags. For
`train` they apply to both `--train` and `--eval`.

| Flag | Meaning |
|---|---|
| `--format {jsonl,json,csv,parquet,hf}` | Input format. Inferred from the file suffix when omitted. |
| `--col-state COLUMN` | Name of the column or field holding the state. Default `state`. |
| `--col-question COLUMN` | Column holding the question. Default `question`. |
| `--col-options COLUMN` | Column holding the options. Default `options`. |
| `--col-answer COLUMN` | Column holding the gold answer. Default `answer_id`. |
| `--col-group COLUMN` | Column holding the group id. Default: `group_id` if present. |
| `--col-id COLUMN` | Column holding the row id. Default: `id` if present. |

These build a [`FieldMapping`](api/data/loaders.md). The mapping's other fields
(`option_ids`, `key`, `option_delimiter`) are available from Python only. See
[Bring your own data](../guides/bring-your-own-data.md).

## `typedecide validate`

Load a dataset and report what is wrong with it.

```text
typedecide validate [-h] [--min-per-class MIN_PER_CLASS] [--json-out JSON_OUT]
                    [input options] source
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | required | Path to a dataset file, or a Hugging Face dataset id |
| `--min-per-class` | `5` | Classes with fewer labelled rows than this get a `rare_class` warning |
| `--json-out` | none | Also write `counts`, `label_balance`, `findings` and `ok` as JSON |

Exits `1` when the report is not OK. Every finding code is explained in
[Read a validation report](../guides/validation-report.md).

```bash
typedecide validate export.csv \
  --col-id "Ticket ID" --col-state "Ticket Body" --col-question "Decision" \
  --col-options "Choices" --col-answer "Chosen" --col-group "Case"
```

## `typedecide split`

Split into train and eval on groups, without leaking a state.

```text
typedecide split [-h] --out OUT [--eval-fraction EVAL_FRACTION] [--seed SEED]
                 [input options] source
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | required | Dataset to split |
| `--out` | required | Directory to write to. Created if missing |
| `--eval-fraction` | `0.2` | Target share of **rows** in eval. Must be strictly between 0 and 1 |
| `--seed` | `0` | Seed for the group shuffle |

Writes `<out>/train.jsonl` and `<out>/eval.jsonl` in the canonical JSONL format,
whatever the input format was. Exits `2` with an explanation if a class would be absent
from eval; see the [runbook](../runbooks/split-refuses.md).

## `typedecide train`

LoRA-tune on the answer token.

```text
typedecide train [-h] --config CONFIG --train TRAIN [--eval EVAL] [input options]
```

| Argument | Default | Meaning |
|---|---|---|
| `--config` | required | YAML or JSON [`TrainConfig`](../guides/training.md) file |
| `--train` | required | Training dataset |
| `--eval` | none | Held-out dataset. When given, eval loss is computed each epoch |

Prints `step N: loss X` as training logs arrive, then the adapter directory, the
manifest path, the final train loss (and eval loss), and the two `evaluate` commands
to run next. All hyperparameters come from the config file; there are no
hyperparameter flags.

## `typedecide evaluate`

Score a model the way a sceptic would.

```text
typedecide evaluate [-h] --model MODEL [--adapter ADAPTER]
                    [--debias {none,cyclic,permutation,calibrated}]
                    [--batch-size BATCH_SIZE] [--device DEVICE] [--seed SEED]
                    [--json-out JSON_OUT] [input options] source
```

| Argument | Default | Meaning |
|---|---|---|
| `source` | required | Labelled dataset to score |
| `--model` | required | Hugging Face model id or local path of the base model |
| `--adapter` | none | LoRA adapter directory to merge before scoring |
| `--debias` | `none` | `cyclic` averages the option rotations, which cancels a position prior. See [the modes](../concepts/position-priors.md#the-four-modes) |
| `--batch-size` | `8` | Prompts per forward pass |
| `--device` | auto | `cuda` when available, otherwise `cpu`. Pass `mps` or another device explicitly |
| `--seed` | `0` | Seeds torch and is recorded in the manifest |
| `--json-out` | none | Also write the metrics and the manifest as JSON |

!!! tip "The CLI default is `none`. The recommendation is `cyclic`."
    `--debias none` scores one ordering and reports `order consistency  n/a`. Use
    `--debias cyclic` unless you are deliberately measuring single-pass behaviour.

## `typedecide export`

Merge an adapter and export ONNX for the browser.

```text
typedecide export [-h] --base BASE [--adapter ADAPTER] --out OUT [--opset OPSET]
                  [--prune-to N] [--keep-merged]
```

| Argument | Default | Meaning |
|---|---|---|
| `--base` | required | Base model id or path |
| `--adapter` | none | LoRA adapter directory. Omit to export the base model |
| `--out` | required | Output directory |
| `--opset` | `17` | ONNX opset. Must be at least 14 |
| `--prune-to N` | off | Prune the unembedding to `N` answer-letter rows. The model can then answer decisions but not generate text. Also writes `slot_map.json` |
| `--keep-merged` | off | Keep the intermediate merged checkpoint in `<out>/_merged` |

This command does not trim logits to the last position and does not quantise. Those are
library calls; see [Export, prune and quantise](../guides/exporting.md).

## `typedecide savings`

What pruning the unembedding would buy, as arithmetic. Loads nothing.

```text
typedecide savings [-h] --vocab VOCAB --hidden HIDDEN [--keep KEEP] [--tied]
                   [--logit-bytes LOGIT_BYTES]
```

| Argument | Default | Meaning |
|---|---|---|
| `--vocab` | required | Vocabulary size (`config.vocab_size`) |
| `--hidden` | required | Hidden size (`config.hidden_size`) |
| `--keep` | `20` | Rows to keep |
| `--tied` | off | The model ties its input embedding to its unembedding. Pruning then saves compute but not download, and `weight_bytes_removed` is reported as 0 |
| `--logit-bytes` | `4` | Bytes per logit: 4 for fp32, 2 for fp16 |

```bash
typedecide savings --vocab 151936 --hidden 1024 --keep 20 --tied
```

```text
  head_parameters                   155,582,464.00
  kept_parameters                   20,480.00
  removed_parameters                155,561,984.00
  weight_bytes_removed              0.00
  logit_bytes_per_position_before   607,744.00
  logit_bytes_per_position_after    80.00
  matmul_flops_per_position_before  311,164,928.00
  matmul_flops_per_position_after   40,960.00

These are arithmetic, not a measurement. Export a pruned model and time it.
```

`weight_bytes_removed` assumes 4.5 bits per weight (the `q4f16` tier). The help text
for `--tied` says every Qwen3 size ties its embeddings; check `tie_word_embeddings` in
your model's `config.json` and do not rely on that sentence.
