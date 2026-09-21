# typedecide

Fine-tune small language models to answer bounded decisions by reading their logits,
and ship the result to a browser.

A *decision* here is one question about one state with a closed set of answers:
*route this ticket, how urgent is it, does this need approval, is this claim
supported*. The conventional approach asks a chat model to emit JSON and parses it
back into control flow. This library does something narrower and more reliable: it
renders the options as lettered slots and takes a softmax over the token ids of those
letters at one position. Nothing is sampled, so a malformed answer is not a thing the
model can produce, and the same input always gives the same output.

```bash
pip install typedecide            # the library and its data tools
pip install typedecide[train]     # + torch, transformers, peft
pip install typedecide[export]    # + optimum, onnx, onnxruntime
```

## Five commands

```bash
typedecide validate tickets.csv                      # what is wrong with your data
typedecide split    tickets.csv --out data/          # without leaking a state
typedecide train    --config train.yaml --train data/train.jsonl
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B --debias cyclic
typedecide export   --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/
```

Or as a library:

```python
from pathlib import Path
from typedecide import load_decisions, validate, group_split
from typedecide.training import TrainConfig, finetune
from typedecide.evaluation import evaluate

decisions = load_decisions("tickets.csv")

report = validate(decisions)
if not report.ok:
    raise SystemExit(report.render())

train, held = group_split(decisions, eval_fraction=0.2, seed=0)
result = finetune(train, TrainConfig(base_model="Qwen/Qwen3-0.6B",
                                     output_dir=Path("runs/lora")))

print(evaluate(held, "Qwen/Qwen3-0.6B", adapter=result.adapter_dir,
               debias="cyclic").render())
```

`import typedecide` pulls in neither torch nor transformers. The module that needs a
heavy dependency imports it when called, and names the extra to install if it is
missing.

## Bring your own data

Point at your columns instead of renaming them first. JSONL, JSON, CSV, Parquet and
Hugging Face dataset ids all load into the same `Decision` type.

```bash
typedecide validate export.csv \
  --col-id "Ticket ID" --col-state "Ticket Body" --col-question "Decision" \
  --col-options "Choices" --col-answer "Chosen" --col-group "Case"
```

CSV options can be a delimited list (`Billing|Shipping|Accounts`, delimiter
auto-detected) or wide columns (`option_1`, `option_2`, …). An answer may be given as
an option id, an answer letter, an index, or the exact option text. Whether a numeric
answer column is 0- or 1-based is decided **once per file**, never row by row; a file
that mixes the two is refused rather than guessed at.

Loading is strict and the errors are diagnoses:

```
error: export.csv row 2, id 'B-1': column 'answer_id' holds 'Accounts', which is not
one of this row's options. It permits the ids billing, shipping (descriptions:
'Billing', 'Shipping'). Fix: use an option id, its answer letter A-B, its 0-based
index, or the exact option text; leave the cell blank for an unlabelled row.
```

`validate` then reports what will hurt you later rather than what merely looks untidy:
contradictory labels (the same state answered two ways), a criterion whose gold is
overwhelmingly one class, option sets that disagree between rows sharing a criterion,
classes too rare to both train on and hold out, states longer than the training
budget, duplicate ids. `report.ok` is false only when the dataset is unusable as given.

## Three things this library is opinionated about

**Split on groups, never on rows.** Every criterion asked about one ticket shares that
ticket's text. Splitting rows puts the same state on both sides and your eval score
becomes a memorisation score. `group_split` splits on `Decision.grouping`, and refuses
with an explanation if the result would leave a class entirely absent from eval.

**Loss lands only on the answer token.** Training builds
`input_ids = encode(prompt) + [answer_token]` and
`labels = [-100] * len(prompt) + [answer_token]`. That one position is the only thing
the readout ever looks at. Teaching the model to write prose it will never be asked
for spends its capacity in the wrong place. Long states are truncated from the *left*,
so the criterion, the options and `Answer:` always survive — truncating the tail would
destroy the answer slot itself.

**Score with `--debias cyclic` unless you have a reason not to.** Small models carry a
strong preference for whichever option happens to be listed first, and plain accuracy
conceals it completely. Averaging the probability of each option *identity* across
rotations of the option list cancels a position prior, because such a prior
contributes equally to every option under a full or cyclic set.

On openjev's `authored144` fixture with Qwen3-0.6B, this is not a small effect:

| Method | Mean family balanced accuracy | Order-unstable rows |
|---|---:|---:|
| chance | 0.333 | — |
| readout, 3 shots | 0.419 | — |
| **readout + cyclic averaging** | **0.483** | 87 / 144 |

87 of 144 decisions changed their answer when the options were reordered. Averaging
recovers 6.4 points and puts a 0.6B model above the 0.440 published for it.

Note what `order_consistency` is telling you though: averaging repairs the *readout*,
not the model. A model that still answers by position and is merely being corrected
afterwards is not the same as a model that reads the evidence. `--debias calibrated`
divides out the prior directly and fixes each individual ordering, which is the better
outcome when it works. Fine-tuning on order-randomised data is the real fix.

## Reproducibility

Every training run writes a `manifest.json` beside its adapter: base model, resolved
config, a content hash of the dataset, row counts, seed, library versions, wall-clock
window, and the resolved answer-slot token ids. A result you cannot trace back to its
input is not a result.

`prompt.py` mirrors `web/prompt.js` byte for byte, and CI fails if they drift.
Training against a prompt your deployment never sends degrades silently and looks
exactly like bad data — it is the most expensive mistake available in this design.

## Making it faster

See [`docs/SPEED.md`](https://ardada2468.github.io/typedecide/SPEED/). The short version: a typed readout looks at one
position and at most twenty token ids, so almost everything the unembedding computes
is discarded. `typedecide savings` does the arithmetic for your model:

```
$ typedecide savings --vocab 151936 --hidden 1024 --keep 20 --tied
  head_parameters                   155,582,464
  removed_parameters                155,561,984
  weight_bytes_removed                        0     <- tied embeddings: no download saving
  logit_bytes_per_position_before        607,744
  logit_bytes_per_position_after              80
  matmul_flops_per_position_before   311,164,928
  matmul_flops_per_position_after         40,960
```

Note the honest row. Qwen3-0.6B, 1.7B and 4B set `tie_word_embeddings=True` (8B and
up do not), so for the sizes that fit a browser the
unembedding *is* the input embedding and pruning it cannot shrink the download — you
still need the full table for lookups. What pruning buys is the matmul and the copy
out of the accelerator. A pruned model answers decisions and cannot generate text.

## Licence

MIT. See [THIRD_PARTY.md](https://github.com/ardada2468/typedecide/blob/main/THIRD_PARTY.md) for the fixtures and prior implementations this work
builds on.
