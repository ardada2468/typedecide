# Tuned model scores worse than base

## Symptoms

`typedecide evaluate ... --adapter runs/lora` reports a lower `balanced accuracy` than
the same command without `--adapter`, or a similar accuracy with a lower
`order consistency`.

## Diagnosis

First make sure the comparison is valid, then look for causes.

**1. Same data, same mode?** Compare the two manifests (`--json-out`):

```bash
python - <<'PY'
import json
a, b = (json.load(open(p))["manifest"] for p in ("eval-base.json", "eval-tuned.json"))
for key in ("fingerprint", "debias", "decisions", "unlabelled_skipped", "seed"):
    print(f"{key:20} {a[key]!s:>40} {b[key]!s:>40}")
PY
```

Any difference in `fingerprint` or `debias` invalidates the comparison.

**2. Is the difference larger than noise?** With `n` eval decisions, the standard
error of an accuracy near `p` is about `sqrt(p * (1 - p) / n)`. At `n = 24` and
`p = 0.5` that is 10 points. A 5-point drop on 24 rows is not a finding.

**3. Does the adapter belong to this base?**

```bash
python -c "import json; m = json.load(open('runs/lora/manifest.json')); print(m['base_model'], m['config']['randomise_option_order'], m['steps'], m['losses'])"
```

`--model` must be the same base the adapter was trained on.

**4. Did training see the same prompt that evaluation sends?** Both use
`typedecide.prompt`, so inside one library version the answer is yes. It is *no* if the
adapter was trained by another tool (for example the prototype scripts in `train/`)
or on a different version of the template.

**5. Does the training distribution match the eval distribution?**

```bash
typedecide validate data/train.jsonl
typedecide validate data/eval.jsonl
```

Compare the `label balance` blocks. Look for `imbalanced` in train: a model tuned on a
criterion that is 90% one class learns the prior, and `balanced_accuracy` punishes
exactly that.

**6. Where is the damage?** Compare `per_criterion` between the two JSON files. One
criterion collapsing is a data problem in that criterion. Every criterion sagging is a
training problem.

**7. Were training states truncated?** Evaluation never truncates; training does. A
model trained on the last 1,000 tokens of a state and evaluated on all 5,000 sees a
different distribution. See the `long_state` warning from `validate`.

## Fix

| Finding | Fix |
|---|---|
| Different fingerprints or modes | Re-run both evaluations identically |
| Eval set too small to tell | Get more held-out data, or report the interval honestly |
| Wrong base model | Evaluate against `manifest["base_model"]` |
| `randomise_option_order` was `false` | Retrain with it `true`. This is the most common cause of a tuned model that looks fine on train order and worse everywhere else |
| Imbalanced training labels | Down-sample the majority class or collect minority examples, then retrain |
| Overfitting (train loss far below eval loss) | Fewer `epochs`, lower `lora_rank`, or more data |
| Underfitting (few steps, loss barely moved) | See [Training loss not decreasing](loss-not-decreasing.md) |
| One criterion collapsed | Check that criterion for `contradictory_labels`, `option_set_mismatch` and `question_mismatch` |
| Truncation mismatch | Raise `max_length`, or shorten states before loading for both training and evaluation |

!!! note "A lower score can be the honest one"
    If the base model was evaluated with `--debias none` on data whose gold answer is
    usually listed first, a position prior **inflates** its accuracy. A tuned model
    that has stopped answering by position can then score lower on that single
    ordering and higher under `--debias cyclic`. Compare under `cyclic`, and compare
    `order_consistency`.

## Verify

- Both evaluation manifests show the same `fingerprint`, `debias` and `decisions`.
- Under `--debias cyclic`, the tuned model's `balanced accuracy` exceeds the base
  model's by more than the noise estimate above.
- The tuned model's `order consistency` is higher than the base model's.
- `per_criterion` shows no criterion below its base value by more than noise.
