# Evaluate like a sceptic

The failure this library's evaluator exists to catch is a model that answers by option
position. On a balanced set that model scores chance, which reads as "weak", not
"broken". So the report always puts three numbers side by side: balanced accuracy,
order consistency, and the letters actually picked.

## Run it

=== "CLI"

    ```bash
    typedecide evaluate data/eval.jsonl \
      --model Qwen/Qwen3-0.6B \
      --adapter runs/lora \
      --debias cyclic \
      --batch-size 8 \
      --json-out eval-tuned.json
    ```

=== "Python"

    ```python
    from pathlib import Path

    from typedecide import load_decisions
    from typedecide.evaluation import evaluate

    held = load_decisions("data/eval.jsonl")
    result = evaluate(held, "Qwen/Qwen3-0.6B", adapter=Path("runs/lora"),
                      debias="cyclic", batch_size=8)
    print(result.render())
    print(result.order_consistency, result.letter_distribution)
    ```

`evaluate` needs the `train` extra. It loads the model with transformers (float32 on
CPU, bfloat16 on any other device), merges the adapter if one is given, and scores in
left-padded batches. The device defaults to `cuda` when available and otherwise
`cpu`; pass `--device mps` or another device string to override.

Unlabelled rows are skipped with a logged warning and counted in the manifest as
`unlabelled_skipped`. If no row is labelled, `DataError` is raised.

## Read the report

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

| Field of `EvalResult` | Meaning | How to read it |
|---|---|---|
| `accuracy` | Plain hit rate of the final prediction | Shown beside `chance`, the mean of 1/n over the rows. Reported for continuity, not for trust |
| `balanced_accuracy` | Mean per-class recall over the classes present in gold, pooled across all criteria | The headline. A constant answer scores 1/k, not the majority share |
| `order_consistency` | Share of decisions whose **per-ordering** argmax was identical under every ordering scored | `None` with `debias="none"`. Low means the answer is a property of the layout |
| `ece` | Expected calibration error, 10 equal-width bins, on the confidence of the final prediction | 0 means stated confidence matches hit rate |
| `brier` | Multi-class Brier score, not halved: 0 is certain and right, 2 is certain and wrong | A uniform guess over k options scores 1 - 1/k |
| `per_criterion` | Balanced accuracy within each criterion key | Find the criterion that is dragging the mean |
| `letter_distribution` | Share of all per-ordering picks that landed on each letter | Under `cyclic`, a model that reads the evidence gives a roughly flat histogram. A spike is a position prior |
| `n` | Labelled decisions scored | |
| `manifest` | How the number was produced | See [Reproducibility](reproducibility.md) |

!!! note "`balanced_accuracy` here is pooled, the bench metric is per family"
    `EvalResult.balanced_accuracy` pools every row and treats each distinct
    `answer_id` string as a class. The `authored144` numbers quoted on this site use
    *mean family balanced accuracy*: balanced accuracy inside each group, then the
    mean over groups. The library provides that as
    `typedecide.evaluation.mean_group_balanced_accuracy`, and the mean of
    `per_criterion` is the same quantity with criteria as the groups. If two criteria
    share option ids such as `yes`/`no`, the pooled figure merges them into one class,
    so prefer `per_criterion` when that applies.

## Choose a debias mode

| Mode | Cost per n-option decision | Use it for |
|---|---:|---|
| `none` | 1 pass | Measuring what production would do with a single pass. Says nothing about position |
| `cyclic` | n passes | The default recommendation. Cancels a slot-only prior and yields `order_consistency` |
| `permutation` | n! passes, refused above 6 options | Auditing a small sample exactly |
| `calibrated` | n passes plus one content-free pass per distinct (question, ordering) | Correcting each individual ordering by dividing out the prior measured with the state `"N/A"` |

`cyclic` is recommended on cost, not on accuracy. On `authored144` cyclic and full
permutation averaging could not be told apart statistically (exact McNemar, p = 0.23),
and both cancel a slot-only prior by the same argument. The mechanics, and the honest
limits of averaging, are in
[Position priors and debiasing](../concepts/position-priors.md).

## The protocol

1. **Score the base model first**, with the same command you will use for the tuned
   model. Without a before number, an after number means nothing. `typedecide train`
   prints both commands when it finishes.
2. **Use the same eval file, the same `--debias`, the same seed.** The manifest's
   `fingerprint` lets you confirm two results were computed on identical data.
3. **Look at `order_consistency` before `balanced_accuracy`.** If it is low, the
   accuracy is being held up by averaging. The model is not reading the evidence yet.
4. **Then run `--debias none`.** That is what a single-pass deployment will do. A gap
   between the `none` and `cyclic` accuracies is the price of the position prior.
5. **Check `per_criterion`.** A good mean can hide one criterion at chance.
6. **Check calibration if you route on confidence.** `ece` and `brier` tell you
   whether a 0.9 means 0.9.
7. **Keep the JSON.** `--json-out` stores the metrics and the manifest.

!!! warning "The eval set must come from a group split"
    If eval shares states with train, every number on this page is inflated. Use
    [`group_split`](../concepts/group-splitting.md), and set `group_id` when several
    rows share a state.

## Score something that is not a local PyTorch model

`evaluate` is `evaluate_with_scorer` with a `TorchReadout` attached. The seam is
`ScoreFn`: any callable that takes a sequence of `ScoreRequest` and returns one
probability vector per request, in order, each as long as that request's option list
and indexed by **position in that request's criterion**. Use it to score an ONNX
export, a remote endpoint, or cached logits with the same debiasing arithmetic.

This example needs no model. The fake scorer always prefers slot A, which is exactly
the pathology the report is designed to show:

```python
from collections.abc import Sequence

from typedecide import load_decisions
from typedecide.evaluation import ScoreRequest, evaluate_with_scorer


def always_first(requests: Sequence[ScoreRequest]) -> list[list[float]]:
    out = []
    for request in requests:
        n = len(request.criterion.options)
        rest = 0.2 / (n - 1)
        out.append([0.8] + [rest] * (n - 1))
    return out


held = load_decisions("data/eval.jsonl")
result = evaluate_with_scorer(held, always_first, debias="cyclic")
print(result.render())
```

On the practice dataset this prints `order consistency  0.0%` and
`letters picked     A:100%  B:0%  C:0%`: a positional answerer, fully exposed. Each
request exposes `request.prompt`, the exact text to score, and `request.criterion`,
the options in the order to present.

A scorer that returns the wrong number of vectors, a vector of the wrong length, a
negative value or a non-finite value is rejected with `ScorerError` (an
`EvaluationError`); see [Handle errors](error-handling.md). A `NaN` is refused
because it would otherwise lose every comparison in the argmax and be scored silently
as the first option.

## Comparing quantisation tiers or exports

Do not use perplexity. A readout compares at most 20 logits and takes an argmax, so
what matters is whether the margin between the top letters survives. Score the same
eval set at each tier through a `ScoreFn` and compare `balanced_accuracy`,
`order_consistency`, then `ece` and `brier`, which usually move first. See
[Making the readout fast](../SPEED.md#3-quantisation-tiers).
