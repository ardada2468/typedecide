# Answer-token-only loss

## One supervised token per example

The readout looks at one position. Training puts the loss on that position and nowhere
else. [`build_example`](../reference/api/training/dataset.md) holds this invariant:

```text
input_ids    = encode(full_prompt(state, criterion)) + [answer_token]
labels       = [-100] * len(encode(full_prompt(...))) + [answer_token]
answer_token = answer_slots(tokenizer, prompt, n_options)[0][decision.answer_index]
```

`-100` is the index PyTorch's cross-entropy ignores, so every prompt token is masked
and the only gradient comes from predicting the answer letter after `Answer:`.

Teaching the model to reproduce ticket text, or to write an explanation it will never
be asked for, spends a small model's capacity on the wrong thing. The answer token is
the one produced by [answer-slot discovery](answer-slots.md), so training supervises
exactly the token the readout will read.

`build_example` only needs an object with an `encode` method, so the invariant is
tested in CI with a stub tokenizer and no torch:

```python
from typedecide import Criterion, Decision, Option
from typedecide.training import IGNORE_INDEX, build_example


class CharTokenizer:
    """One token per character. Enough to see the masking."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


criterion = Criterion("queue", "Which queue?", (
    Option("billing", "Billing"), Option("shipping", "Shipping"),
))
decision = Decision(id="T-1", state="Charged twice.", criterion=criterion,
                    answer_id="shipping")

example = build_example(CharTokenizer(), decision, max_length=1024)
supervised = [t for t in example["labels"] if t != IGNORE_INDEX]
print(supervised, chr(supervised[0]))     # [66] B
```

## Truncation comes from the left of the state

When `encode(prompt)` plus the answer token exceeds `max_length`, something has to go.
Ordinary right-truncation would cut off the criterion, the options and `Answer:`. That
destroys the answer slot: the supervised position would land on a token that is not an
answer letter.

So `build_example` removes tokens from the **left of the state**:

1. The tail (criterion, options, `Answer:`) is protected and never touched.
2. The instruction preamble (`INSTRUCTION` plus `Evidence:`) is kept at the front when
   there is room.
3. The oldest part of the state is dropped, keeping the evidence nearest the
   criterion.

Edge cases, all handled explicitly:

| Situation | Behaviour |
|---|---|
| The tail alone needs more than `max_length - 1` tokens | `TrainingError`: raise `max_length` or shorten the option descriptions |
| No room for the preamble and any evidence | Warning logged; only the state tokens nearest the criterion are kept |
| The tokenizer does not encode the head as a token-level prefix of the prompt | Warning logged; the whole prompt is truncated from the left, which still preserves the tail |

!!! warning "Evaluation never truncates"
    `TorchReadout` scores over-length prompts in full and logs a warning when a
    `max_length` was given, because truncating at scoring time would move or destroy
    the answer boundary. A state that was truncated in training is seen whole at
    evaluation. If that mismatch matters to you, shorten or summarise states before
    loading.

!!! note "`validate` warns about long states using an estimate"
    The `long_state` finding uses a tokenizer-free estimate (the larger of about 1.3
    tokens per word and one token per four characters) against a 1024-token budget,
    so validation never downloads a model. It is a smoke alarm. Its message says the
    *end* of the ticket is what gets lost; for training in this library it is the
    **start** of the state that is dropped, as described above.

## Padding: right for training, left for scoring

The two are different on purpose.

- **Training** right-pads (`pad_batch`). Real tokens keep positions `0..n-1`, which is
  what default position ids give, and padding carries `-100` labels and a zero
  attention mask.
- **Scoring** left-pads and passes explicit
  `position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`. The readout takes
  the logits at index `-1`; with right padding that index is a pad token for every
  row except the longest. And without explicit position ids, a left-padded row is
  told its first real token sits at position `gap`, so a row's answer would change
  depending on which other prompts shared its batch.

Both mistakes produce wrong numbers silently, not a crash, which is why they are
spelled out in the source.
