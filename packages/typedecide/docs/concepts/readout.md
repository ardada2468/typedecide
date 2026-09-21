# The typed logit readout

## The mechanism

For a state and a criterion, [`full_prompt`](../reference/api/prompt.md) renders:

```text
Apply the criterion to the evidence and choose exactly one listed option. Answer with only the uppercase letter of that option.

Evidence:
I was charged twice for order A-9923.

Criterion: Which queue should handle this ticket?
Options:
A. Billing and payments
B. Shipping and delivery
C. Account access and authentication
Answer:
```

The model runs one forward pass over that text. At the final position it produces a
logit for every token in its vocabulary. The readout keeps the logits of the tokens
that encode `A`, `B` and `C` at that boundary, and takes a softmax over just those:

```text
logits[-1][slot(A)], logits[-1][slot(B)], logits[-1][slot(C)]  -> softmax -> p(billing), p(shipping), p(access)
```

In [`TorchReadout`](../reference/api/evaluation/readout.md) that is literally
`torch.softmax(logits[row, slots], dim=-1)`.

You can build the prompt yourself without any model:

```python
from typedecide import Criterion, Option, full_prompt

criterion = Criterion(
    key="queue",
    question="Which queue should handle this ticket?",
    options=(
        Option(id="billing", description="Billing and payments"),
        Option(id="shipping", description="Shipping and delivery"),
    ),
)
print(full_prompt("I was charged twice for order A-9923.", criterion))
```

## What this buys

- **A malformed answer is structurally impossible.** The softmax ranges over this
  criterion's own letters. An option belonging to another criterion, a missing key,
  or an unclosed brace are not values it can take.
- **Determinism.** Nothing is sampled, so the same input gives the same output.
- **A probability per option.** That is what makes averaging over orderings,
  calibration, ECE and Brier possible, and what lets an application route
  low-confidence decisions to a person.

## What it does not buy

**Accuracy.** On `authored144` with Qwen3-0.6B, greedy generation of the answer
letter and the readout score the same 0.419 mean family balanced accuracy
(`bench/results/*-generate-shots3.json` and `*-readout-shots3.json`), because greedy
decoding picks the argmax the readout reads. The case for the readout is the list
above, not a better score.

## Why letters and not option text

One could score each option by the likelihood of its description text instead. The
library scores letters for four reasons:

1. **One token, one position.** Every letter is a single token at the same position,
   so the scores are directly comparable. Option descriptions differ in length and
   in how common their words are, which needs a length normalisation with no single
   right answer.
2. **One forward pass per criterion.** Scoring option text needs a pass, or at least a
   separate continuation, per option.
3. **Ids stay out of the prompt.** The model reads `A. Billing and payments`. Your
   code receives `billing`. You can rename ids freely without changing a single
   prompt token, and the model never sees an internal identifier.
4. **It makes the head prunable.** Because only the letter rows of the unembedding are
   ever read, the other rows can be removed at export time. See
   [Making the readout fast](../SPEED.md).

The cost of choosing letters is that a letter is a *position*, and small models have
opinions about positions. That is the subject of
[Position priors and debiasing](position-priors.md).

## Why 20 options at most

`LETTERS` is `"ABCDEFGHIJKLMNOPQRST"`. Twenty is the ceiling because that is how many
letters are rendered; beyond that a tokenizer is unlikely to give one clean token per
label. `Criterion` refuses fewer than 2 or more than 20 options with a `SchemaError`.

## The answer is an id, never a position

`Decision.answer_id` names an option. `Criterion.reordered(order)` permutes the
options and the answer stays correct without being touched:

```python
from typedecide import Criterion, Decision, Option

criterion = Criterion("queue", "Which queue?", (
    Option("billing", "Billing"), Option("shipping", "Shipping"), Option("access", "Access"),
))
decision = Decision(id="T-1", state="Charged twice.", criterion=criterion, answer_id="shipping")
print(decision.answer_index)                                   # 1

rotated = Decision(id="T-1", state="Charged twice.",
                   criterion=criterion.reordered([2, 0, 1]), answer_id="shipping")
print(rotated.answer_index)                                    # 2
```

Everything that permutes options (training-time randomisation, evaluation-time
debiasing) depends on this.
