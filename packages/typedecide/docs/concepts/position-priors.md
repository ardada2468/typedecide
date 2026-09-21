# Position priors and debiasing

## The problem

A small model asked to answer with a letter has a preference for certain letters
before it reads any evidence. If the preference is strong, the model is answering by
*position*. On a balanced dataset such a model scores about chance, and chance in an
accuracy column looks like a weak model, not a broken one.

The measured size of the effect, on `authored144` with Qwen3-0.6B
(`bench/results/authored144-Qwen3-0.6B-ONNX-q4f16-readout-cyclic-shots3.json`):
**87 of 144 decisions changed their argmax when the options were rotated.** The
single-ordering runs of the same model report 0 unstable rows, only because they never
looked.

## The three numbers that expose it

[`evaluate`](../reference/api/evaluation/runner.md) always reports these together:

| Number | Question it answers |
|---|---|
| `balanced_accuracy` | Is it right, with every class weighted equally? |
| `order_consistency` | Of the decisions scored under more than one ordering, what share got the **same** answer under every ordering? `None` when only one ordering was scored. |
| `letter_distribution` | Which letters did it actually pick? `A:100% B:0% C:0%` is a complete diagnosis. |

`order_consistency` is computed from the **per-ordering** argmaxes, before any
averaging. Debiasing therefore cannot hide a positional model from it.

## The four modes

`debias` accepts `"none"`, `"cyclic"`, `"permutation"` or `"calibrated"`.

| Mode | Orderings scored per decision | Forward passes for an n-option decision | What is done with them |
|---|---|---:|---|
| `none` | The order given | 1 | Argmax. `order_consistency` is `None`. |
| `cyclic` | The n rotations | n | Average the probability per **option id**, then argmax. |
| `permutation` | All n! arrangements | n! | Same averaging. Refused above 6 options (`ConfigError`). |
| `calibrated` | The n rotations | n, plus one content-free pass per distinct (question, ordering) | Divide each vector by the content-free prior and renormalise, then average per option id. |

### Why averaging cancels a position prior

Suppose the model's score for an option is its real evidence for that option plus a
bonus that depends only on the slot it sits in. Under a cyclic set, every option sits
in every slot exactly once. Each option therefore collects the same total bonus. When
you average per option id, the bonus is a constant added to every option and it drops
out of the argmax. The same holds for the full permutation set.

The accumulator is keyed by option **id**, never by position. Position is the thing
being forgotten.

### What `calibrated` does

It measures the prior directly. For each distinct (criterion key, question, option
order) it scores the same prompt with the state replaced by the content-free string
`"N/A"` (`NULL_STATE`), caches that distribution, divides every real probability
vector by it and renormalises. Unlike averaging, this adjusts **each individual
ordering**, so when it works the single-ordering answer itself becomes right.

## Averaging repairs the readout, not the model

This is the point most worth being honest about.

After cyclic averaging, the *reported prediction* no longer depends on the option
order. The *model* is unchanged. It still flips its answer when the options move, and
`order_consistency` still says so. A model that reads the evidence and a model that
answers by position and is corrected afterwards are different things:

- The corrected model needs n forward passes per decision in production, forever.
- Its per-ordering probabilities are not trustworthy as confidences.
- The correction assumes the bias is a function of slot alone. A bias that interacts
  with content does not cancel.

The measured numbers make the same point. Averaging moved balanced accuracy from 0.419
to 0.483 while 87 of 144 rows were order-unstable. The improvement and the instability
are both real.

!!! warning "Cyclic and full permutation averaging cannot be ranked on this evidence"
    On `authored144`, full permutation averaging (6 orderings for 3 options) scored
    0.442 with 106 of 144 rows unstable, and cyclic averaging (3 orderings) scored
    0.483 with 87. That difference is **not statistically distinguishable**. The two
    runs disagree on 11 rows (cyclic right on 8, full permutation right on 3), and an
    exact McNemar test on those discordant pairs gives a two-sided p = 0.23
    (`bench/compare.mjs`). The mechanism predicts no difference either: both sets put
    every option in every position equally often. Do not conclude that fewer orderings
    are better. What *is* established is cost: cyclic needs n passes and full
    permutation needs n!, and that is the reason this site recommends cyclic as the
    default.

    The `calibrated` figure in the same results (0.459, 80 of 144 unstable) is
    contextual calibration applied to each of the 3 cyclic rotations and then
    averaged, which is also what `debias="calibrated"` does in this library. It has
    not been tested against the other modes for significance. These are single runs on
    one fixture with one model; measure on your own data.

## The real fix

Fine-tune on order-randomised data. `TrainConfig.randomise_option_order` is `True` by
default and re-permutes every decision's options at the start of **every epoch**, so
"the answer is A" is never a hypothesis consistent with the training data. The eval
set is never re-permuted during training, because a moving eval set is not a
measurement.

After tuning, the success criterion is `order_consistency` close to 1.0 with
`debias="cyclic"`, and `debias="none"` accuracy close to the debiased accuracy. When
those hold, you can drop the extra passes in production.

## Doing it by hand

The augmentation functions are public, for audits on a sample:

```python
from typedecide import Criterion, Decision, Option
from typedecide.data import all_permutations, all_rotations, randomise_option_order

criterion = Criterion("queue", "Which queue?", (
    Option("billing", "Billing"), Option("shipping", "Shipping"), Option("access", "Access"),
))
decision = Decision(id="T-1", state="Charged twice.", criterion=criterion, answer_id="billing")

for variant in all_rotations(decision):
    print(variant.id, variant.criterion.option_ids, variant.answer_id)
# T-1#r0 ('billing', 'shipping', 'access') billing
# T-1#r1 ('shipping', 'access', 'billing') billing
# T-1#r2 ('access', 'billing', 'shipping') billing

print(len(all_permutations(decision)))                       # 6
shuffled = randomise_option_order([decision], seed=0)        # same ids, same answers
```

`all_permutations` refuses more than 8 options (8! is 40,320 variants of one row).
`randomise_option_order` seeds each row from `(seed, decision.id)`, so a row gets the
same ordering whichever dataset it is part of and whatever order it was loaded in.
