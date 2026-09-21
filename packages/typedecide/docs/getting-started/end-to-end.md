# End to end

This page runs every stage on the practice dataset from
[Your first dataset](first-dataset.md). The validate and split output below is the
real output of those commands on that file. Training, evaluation and export output
depends on your hardware and is described rather than quoted.

## 1. Validate

```bash
typedecide validate tickets.jsonl
```

```text
INFO typedecide.data.loaders: loaded 120 decisions from tickets.jsonl
INFO typedecide.data.validate: validated 120 decisions: 0 error(s), 0 warning(s)
Dataset report: OK
  decisions=120, labelled=120, unlabelled=0, criteria=2, groups=60, unique_states=60, errors=0, warnings=0
  -- label balance --
    queue: access=20 (33%), billing=20 (33%), shipping=20 (33%)
    refund_requested: no=30 (50%), yes=30 (50%)
  Nothing to report.
```

The exit status is 0 when the report is OK and 1 when any finding has severity
`error`, so the command works as a CI gate. Real data is rarely this clean;
[Read a validation report](../guides/validation-report.md) explains each finding.

## 2. Split

```bash
typedecide split tickets.jsonl --out data/
```

```text
120 decisions in 60 groups
  data/train.jsonl  96 decisions
  data/eval.jsonl   24 decisions

Split on groups, not rows, so no state appears on both sides.
```

The split is deterministic given `--seed` (default 0) and does not depend on the order
rows were loaded in.

!!! warning "A small dataset can make the split refuse"
    `group_split` raises `DataError` if any `(criterion, answer)` class present in the
    data would have zero examples in eval. With a handful of rows that is the likely
    outcome. The message lists the missing classes and how many groups each spans.
    See the [runbook](../runbooks/split-refuses.md).

## 3. Write a training config

```yaml title="train.yaml"
base_model: Qwen/Qwen3-0.6B
output_dir: runs/lora        # relative paths resolve against this file's directory
epochs: 2
batch_size: 4
grad_accum: 4
learning_rate: 1.0e-4
lora_rank: 16
max_length: 1024
seed: 0
randomise_option_order: true
```

Only `base_model` and `output_dir` are required. Unknown keys are rejected with a list
of the permitted ones. Every field is documented in
[Configure training](../guides/training.md).

## 4. Train

```bash
pip install "typedecide[train]"
typedecide train --config train.yaml --train data/train.jsonl --eval data/eval.jsonl
```

What happens, in order:

1. Every training row is checked for an `answer_id`. **Unlabelled training rows stop
   the run** with a `TrainingError` naming the first five. Unlabelled *eval* rows are
   dropped with a logged warning.
2. The tokenizer is loaded and the answer-letter token ids are resolved once per
   distinct criterion (the *preflight*). A tokenizer with no stable answer slot fails
   here, in seconds, not at step 4,000.
3. The base model is wrapped with LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `gate_proj`, `up_proj` and `down_proj`.
4. Training runs with a cosine schedule and 3% warmup. Option order is re-randomised
   at the start of every epoch. Loss is logged every 25 steps and the CLI prints it as
   `step N: loss X`.
5. The adapter, the tokenizer and `manifest.json` are written to `output_dir`.

The command finishes by printing the adapter path, the manifest path, the final
losses, and the two evaluation commands to run next.

!!! note "The loss is a single-token loss"
    It is the cross-entropy of one answer letter per example, over the **whole
    vocabulary**. With three options, a model that has learned the output format but
    nothing else sits near `ln 3 = 1.10`. Read it against that, not against typical
    language-model losses.

## 5. Evaluate, before and after, the same way

```bash
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B --debias cyclic
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B --adapter runs/lora --debias cyclic
```

The report has this shape (the values here are placeholders, not a result):

```text
model              Qwen/Qwen3-0.6B + runs/lora
decisions          24
debias             cyclic   3 orderings scored
accuracy           __._%   (chance 41.7%)
balanced accuracy  __._%
order consistency  __._%   same answer under every ordering
ece                _.___
brier              _.___
letters picked     A:__%  B:__%  C:__%

per criterion (balanced accuracy):
  queue                _.___
  refund_requested     _.___
```

Read three lines together: **balanced accuracy** (is it right), **order consistency**
(does the answer survive reordering the options) and **letters picked** (a histogram
that reads `A:100%` is the whole diagnosis). The reasoning is in
[Evaluate like a sceptic](../guides/evaluating.md).

Add `--json-out result.json` to keep the numbers and the run manifest.

## 6. Export

```bash
pip install "typedecide[export]"
typedecide export --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/
```

This merges the adapter into the base weights, exports a decoder with a past-key
cache through Optimum (task `text-generation-with-past`, opset 17), and writes
`export_manifest.json`. Add `--prune-to N` to slice the LM head down to `N`
answer-letter rows, which also writes `slot_map.json`.

!!! danger "A pruned model cannot generate text"
    It has output rows for the answer letters and nothing else. Keep the unpruned
    checkpoint. Also note that the browser demo in `web/` reads logits by vocabulary
    id and does not yet consume `slot_map.json`, so it cannot load a pruned export.
    Details in [Export, prune and quantise](../guides/exporting.md).

Trimming the graph to the last position and quantising are library calls, not CLI
flags:

```python
from typedecide.export import quantize, trim_logits_to_last_position

trim_logits_to_last_position("onnx/")
quantize("onnx/", mode="q4f16")
```

## The same pipeline in Python

```python
from pathlib import Path

from typedecide import group_split, load_decisions, validate
from typedecide.data import write_decisions
from typedecide.evaluation import evaluate
from typedecide.export import ExportConfig, export_onnx
from typedecide.training import TrainConfig, finetune

decisions = load_decisions("tickets.jsonl")

report = validate(decisions)
print(report.render())
if not report.ok:
    raise SystemExit(1)

train, held = group_split(decisions, eval_fraction=0.2, seed=0)
write_decisions(train, Path("data/train.jsonl"))
write_decisions(held, Path("data/eval.jsonl"))

config = TrainConfig(base_model="Qwen/Qwen3-0.6B", output_dir=Path("runs/lora"))
result = finetune(train, config, eval_set=held)

before = evaluate(held, config.base_model, debias="cyclic")
after = evaluate(held, config.base_model, adapter=result.adapter_dir, debias="cyclic")
print(before.render(), after.render(), sep="\n\n")

export_onnx(ExportConfig(
    base_model=config.base_model,
    adapter=result.adapter_dir,
    output_dir=Path("onnx"),
))
```

Next: the [concepts](../concepts/index.md) explain why each stage is built the way it
is, and the [how-to guides](../guides/index.md) cover each stage in depth.
