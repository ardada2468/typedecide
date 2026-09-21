# Answer-slot discovery

The readout needs the token id that each answer letter encodes to *at the position
right after `Answer:`*. There are two tempting shortcuts, and both are wrong.

**"The letters are contiguous, so use `id(A) + i`."** They are not contiguous in any
vocabulary this project has checked. `web/models.json` says the same thing in its own
header note: a hard-coded `labelBase + index` is silently wrong.

**"Just encode `"A"`."** Whether the natural continuation after `Answer:` is `" A"`
(with a leading space) or `"A"` depends on the tokenizer and on the character before
it. The token for `" A"` and the token for `"A"` are different ids. Reading the wrong
one scores a token the model rarely emits there.

## The probe

[`answer_slots(tokenizer, prompt, count)`](../reference/api/prompt.md) does this:

1. Encode the prompt. Call the result `base`.
2. For each separator in `(" ", "")`, and for each letter, encode
   `prompt + separator + letter`.
3. Accept the separator only if **every** letter appends exactly one token and leaves
   the first `len(base)` tokens unchanged, and the resulting ids are all distinct.
4. Return `(slots, separator)` for the first separator that passes.

If neither separator passes, it raises
[`PromptError`](../reference/api/errors.md):

```text
No answer-letter continuation keeps the prompt boundary stable for this tokenizer, so
a readout would score tokens that are not the answer. The prompt tail needs adjusting
for this model.
```

That is a deliberate stop. The alternative is a readout that runs and returns numbers
that mean nothing. The
[runbook](../runbooks/prompt-error-tokenizer.md) covers what to do about it.

The tokenizer argument only needs an `encode(text, add_special_tokens=False)` method
(the `Tokenizer` protocol), so you can probe any Hugging Face tokenizer directly:

```python
from transformers import AutoTokenizer

from typedecide import Criterion, Option, answer_slots, full_prompt

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
criterion = Criterion("probe", "Which option applies?", (
    Option("a", "First"), Option("b", "Second"), Option("c", "Third"),
))
slots, separator = answer_slots(tokenizer, full_prompt("Some evidence.", criterion), 3)
print(slots, repr(separator))
```

## Where the probe runs

| Place | When | On failure |
|---|---|---|
| `DecisionDataset.preflight()` | Once per distinct criterion, before training starts | `TrainingError` naming the decision and criterion |
| `build_example()` | Per training example | `TrainingError` naming the decision |
| `TorchReadout` | Per distinct prompt during scoring, cached | `PromptError` |
| `answer_letter_token_ids()` | At export, against a neutral probe prompt, to choose the rows to keep when pruning | `ExportError` if the tokenizer cannot load, `PromptError` if no separator is stable |
| `web/engine.js` `answerSlots()` | In the browser, per criterion | A thrown `Error` with the same wording |

The resolved ids are written to the training manifest under `answer_slots`, keyed by
criterion and letter, so a result can be traced to the exact tokens it supervised.

!!! note "The separator matters downstream"
    A pruned export records the separator in `slot_map.json`. If the export cannot
    reproduce the kept ids by probing, it records `null` and logs a warning: the
    readout and the export may be scoring different tokens.
