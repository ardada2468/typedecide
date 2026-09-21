# Training loss not decreasing

## Symptoms

- `step N: loss X` stays flat from the first log to the last.
- The loss sits near `ln(number of options)`: about 0.69 for 2 options, 1.10 for 3,
  1.39 for 4.
- The loss is `nan`, or jumps upward and stays there.
- The loss falls to almost zero within the first epoch.

!!! note "What the loss is"
    One token per example is supervised, and the cross-entropy is over the **whole
    vocabulary**. `ln(n)` therefore means "the model has learned to emit an answer
    letter, and is guessing which". A value far above `ln(n)` means it has not yet
    learned that a letter comes next. Loss is logged every 25 optimiser steps, so a
    small dataset produces very few log lines.

## Diagnosis

Work through these in order. Each is cheaper than the next.

**1. Is there anything to learn?**

```bash
typedecide validate data/train.jsonl
```

Look for `contradictory_labels` (a hard ceiling), `imbalanced` (the prior is the
cheapest hypothesis) and `rare_class`.

**2. How many optimiser steps did the run take?**

```bash
python -c "import json; m = json.load(open('runs/lora/manifest.json')); print(m['steps'], m['config']['effective_batch_size'], m['dataset']['train']['rows'])"
```

Steps are roughly `rows * epochs / effective_batch_size`. With the defaults
(effective batch 16, 2 epochs), 96 rows give 12 steps. The 3% warmup and cosine decay
over 12 steps leave very little training.

**3. Are examples being truncated?** Run with `-v` and look for truncation messages:

```bash
typedecide -v train --config train.yaml --train data/train.jsonl 2>&1 | grep -i truncat
```

`state truncated from the left` is expected for long rows. `max_length leaves no room
for the instruction preamble` and `could not locate the state/criterion token
boundary` mean `max_length` is too small for your criteria.

**4. Is the supervised token the right one?** Check the manifest's `answer_slots`
against the tokenizer:

```python
import json

from transformers import AutoTokenizer

manifest = json.load(open("runs/lora/manifest.json"))
tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"])
for key, slots in manifest["answer_slots"].items():
    print(key, {letter: tokenizer.convert_ids_to_tokens([i])[0] for letter, i in slots.items()})
```

Every value should render as the letter itself, possibly with a leading-space marker.

**5. Is the learning rate sane for this model?** `nan` or a jump upward with
`learning_rate` above the default `1e-4` points here. On CPU the model trains in
float32; on CUDA in bfloat16.

## Fix

| Cause | Change |
|---|---|
| Too few optimiser steps | Raise `epochs`, or lower `grad_accum` so the same data gives more steps. Keep `batch_size * grad_accum` at 8 or more |
| Loss stuck at `ln(n)` with enough steps | The evidence may not determine the answer. Read twenty rows yourself. Fix `contradictory_labels`; add the distinguishing text to the state; check that option descriptions differ meaningfully |
| Loss far above `ln(n)` and flat | The adapter is not learning at all. Raise `learning_rate` towards `2e-4` or `lora_rank` to 32, one change at a time |
| `nan` or divergence | Lower `learning_rate` by 3x to 10x |
| Heavy truncation | Raise `max_length`, or summarise states before loading |
| Loss near zero immediately | Usually a dataset a model can solve from surface cues, or duplicated rows. Check `duplicate_rows`, then trust only the held-out evaluation |

Do not turn off `randomise_option_order` to make the loss fall faster. It will fall
faster, because "answer A" becomes learnable, and the model will be
[answering by position](answers-by-position.md).

## Verify

- A rerun shows the logged loss falling below `ln(n_options)` for your criteria.
- With `--eval`, the eval loss in the final line falls too. A falling train loss with a
  rising eval loss is overfitting; reduce `epochs`.
- The number that matters is downstream:

    ```bash
    typedecide evaluate data/eval.jsonl --model <base> --debias cyclic
    typedecide evaluate data/eval.jsonl --model <base> --adapter runs/lora --debias cyclic
    ```

    The tuned run should beat the base run on `balanced accuracy` **and** on
    `order consistency`.
