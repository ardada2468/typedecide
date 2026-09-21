# typedecide

Fine-tune small language models to answer bounded decisions by reading their logits,
and ship the result to a browser.

A *decision* here is one question about one piece of evidence (the *state*) with a
closed set of answers: *route this ticket*, *how urgent is it*, *does this need
approval*, *is this claim supported*. The usual approach asks a chat model to write
JSON and parses it back into control flow. `typedecide` does something narrower: it
renders the options as lettered slots (`A.`, `B.`, `C.` ...) and takes a softmax over
the token ids of those letters at the single position after `Answer:`.

Nothing is sampled. A malformed answer is not something the model can produce, the
same input always gives the same output, and every option comes back with a
probability.

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Read the concepts](concepts/index.md){ .md-button }
[Try the browser demo](demo/){ .md-button }

## What is in the box

| Stage | What it does | Entry point |
|---|---|---|
| Load | JSONL, JSON, CSV/TSV, Parquet or a Hugging Face dataset id into one `Decision` type, mapping your column names rather than renaming them | [`load_decisions`](reference/api/data/loaders.md) |
| Validate | Reports contradictory labels, imbalance, mismatched option sets, rare classes, long states, duplicate ids | [`validate`](reference/api/data/validate.md) |
| Split | Splits on groups, never rows, and refuses if a class would vanish from eval | [`group_split`](reference/api/data/split.md) |
| Train | LoRA fine-tune where the loss lands on the answer-letter token only | [`finetune`](reference/api/training/trainer.md) |
| Evaluate | Balanced accuracy, order consistency, letter distribution, ECE, Brier; four debiasing modes | [`evaluate`](reference/api/evaluation/runner.md) |
| Export | Merge the adapter, optionally prune the LM head to the answer letters, export ONNX, trim, quantise | [`export_onnx`](reference/api/export/onnx.md) |

`import typedecide` pulls in neither torch nor transformers. A function that needs a
heavy dependency imports it when called, and the error names the extra to install if
it is missing.

## Five commands

```bash
typedecide validate tickets.csv                      # what is wrong with your data
typedecide split    tickets.csv --out data/          # without leaking a state
typedecide train    --config train.yaml --train data/train.jsonl
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B --debias cyclic
typedecide export   --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/
```

The same thing as a library:

```python
from pathlib import Path

from typedecide import group_split, load_decisions, validate
from typedecide.evaluation import evaluate
from typedecide.training import TrainConfig, finetune

decisions = load_decisions("tickets.csv")

report = validate(decisions)
if not report.ok:
    raise SystemExit(report.render())

train, held = group_split(decisions, eval_fraction=0.2, seed=0)
result = finetune(
    train,
    TrainConfig(base_model="Qwen/Qwen3-0.6B", output_dir=Path("runs/lora")),
    eval_set=held,
)

print(evaluate(held, "Qwen/Qwen3-0.6B", adapter=result.adapter_dir,
               debias="cyclic").render())
```

The [end-to-end walkthrough](getting-started/end-to-end.md) runs every step on a
dataset you can generate in ten seconds.

## The measured headline

Every number below is read from a committed result file in
[`bench/results/`](https://github.com/ardada2468/typedecide/tree/main/bench/results).
Fixture: openjev's `authored144` (144 authored decisions, 3 families, 3 options each,
so chance is 0.333). Model: `onnx-community/Qwen3-0.6B-ONNX` at `q4f16`. Metric: mean
family balanced accuracy.

| Method | Shots | Balanced accuracy | Order-unstable rows | Run time (s) |
|---|---:|---:|---:|---:|
| chance | | 0.333 | | |
| readout | 0 | 0.385 | not checked | 56.8 |
| readout | 3 | 0.419 | not checked | 189.6 |
| generation | 3 | 0.419 | not checked | 225.3 |
| readout + full permutation averaging | 3 | 0.442 | 106 / 144 | 1102.2 |
| readout + contextual calibration (per rotation, then averaged) | 3 | 0.459 | 80 / 144 | 918.2 |
| **readout + cyclic averaging** | 3 | **0.483** | 87 / 144 | 537.3 |

Three things to take from it:

1. **Generation equals readout at 0.419.** Greedy decoding of an answer letter picks
   the same argmax the readout reads. The readout does not buy accuracy. It buys a
   structural guarantee, determinism, and a probability per option.
2. **87 of 144 decisions changed their answer when the options were reordered.** At
   this model size the position prior is the dominant error source, and plain
   accuracy does not show it. "Not checked" in the table is literal: a run that
   scores one ordering has no way to see the problem.
3. **Cyclic averaging recovers 6.4 points (0.419 to 0.483).** It repairs the
   *readout*, not the model. See
   [Position priors and debiasing](concepts/position-priors.md).
4. **Cyclic and full permutation averaging cannot be ranked.** They disagree on 11
   rows (cyclic right on 8, full permutation right on 3); an exact McNemar test gives
   p = 0.23. Cyclic is highlighted because it costs n passes where full permutation
   costs n!, not because it is shown to be more accurate.

!!! warning "Where these numbers come from, and where they do not"
    They were produced by the JavaScript bench harness (`bench/authored144.mjs`,
    ONNX Runtime, 3 in-context examples), which shares the prompt and the debiasing
    arithmetic with this library. They were **not** produced by `typedecide evaluate`,
    which scores a PyTorch model with no in-context examples. Expect the same
    qualitative picture from the Python path; do not expect the same digits. No
    fine-tuned result is committed yet, so this site quotes none.

## Where to go next

<div class="grid cards" markdown>

- **New here?** Start with [Install](getting-started/install.md), then
  [Your first dataset](getting-started/first-dataset.md).
- **Have a CSV from a ticketing system?** Go to
  [Bring your own data](guides/bring-your-own-data.md).
- **Something broke?** The [runbooks](runbooks/index.md) are organised by symptom.
- **Deciding what to optimise?** Read
  [Making the readout fast](SPEED.md). Every claim in it is labelled MEASURED,
  DERIVED or PROJECTED.

</div>
