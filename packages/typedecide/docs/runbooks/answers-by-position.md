# Model answers by position (low order_consistency)

## Symptoms

From `typedecide evaluate ... --debias cyclic`:

```text
order consistency  12.5%   same answer under every ordering
letters picked     A:81%  B:13%  C:6%
```

- `order consistency` is low.
- One letter dominates `letters picked`. Under `cyclic` every option visits every
  slot, so a model that reads the evidence gives a roughly flat histogram.
- Accuracy under `--debias none` is near chance, or, if your gold answers are usually
  listed first, suspiciously high.
- In the browser demo, ticking *Shuffle option order* changes the answers.

For scale: stock Qwen3-0.6B on `authored144` changed its answer on 87 of 144 decisions
under rotation. This is the expected state of a small untuned model, not an exotic
failure.

## Diagnosis

**1. Measure it properly.** `--debias none` cannot show this; it reports
`order consistency  n/a`.

```bash
typedecide evaluate data/eval.jsonl --model <base> --debias cyclic --json-out base-cyclic.json
typedecide evaluate data/eval.jsonl --model <base> --adapter runs/lora --debias cyclic --json-out tuned-cyclic.json
```

**2. Is the bias in the base model, the adapter, or both?** Compare
`order_consistency` and `letter_distribution` across the two files. If the adapter is
*worse* than the base, training introduced it.

**3. Was option order randomised during training?**

```bash
python -c "import json; m = json.load(open('runs/lora/manifest.json')); print(m['option_order_randomisation'])"
```

`per-epoch` is correct. `off` is the cause.

**4. Is the gold position skewed in the training file?** With randomisation on this
does not matter; with it off it is fatal.

```python
from collections import Counter

from typedecide import load_decisions

rows = [d for d in load_decisions("data/train.jsonl") if d.labelled]
print(Counter(d.answer_index for d in rows))
```

## Fix

**The real fix: fine-tune with order randomisation.**

```yaml title="train.yaml"
randomise_option_order: true     # the default; make sure nothing set it to false
```

Every decision's options are re-permuted at the start of each epoch, so position
carries no information about the answer. More epochs mean more distinct orderings per
decision.

**If randomisation was already on** and consistency is still low:

- Train longer or on more data. Check the step count in the manifest; a run of a
  dozen steps will not move a prior.
- Check for [a loss stuck at `ln(n)`](loss-not-decreasing.md). A model that cannot
  extract signal from the evidence falls back to position.
- Make option descriptions more distinct. Near-identical descriptions leave position
  as the tiebreaker.

**The interim fix: debias at inference.**

- `cyclic` costs n passes for n options and cancels a slot-only prior in the averaged
  prediction. In the browser each extra ordering is one short suffix against the
  prefilled state, so this is affordable there.
- `calibrated` divides out a content-free prior for each rotation, then averages. On
  `authored144` it scored 0.459 with 80 of 144 rows unstable, against 0.483 and 87
  for plain cyclic averaging. No significance test has been run on that pair, so do
  not read it as a ranking; try both on your own data.

!!! warning "Debiasing repairs the readout, not the model"
    With averaging in place the reported answer is stable, and `order_consistency`
    stays low, because it is computed from the per-ordering answers. Keep reporting
    it. A deployment that depends on averaging pays n passes per decision for as long
    as it runs, and its single-ordering confidences are not trustworthy. See
    [Position priors and debiasing](../concepts/position-priors.md#averaging-repairs-the-readout-not-the-model).

## Verify

After retraining:

```bash
typedecide evaluate data/eval.jsonl --model <base> --adapter runs/lora --debias cyclic
typedecide evaluate data/eval.jsonl --model <base> --adapter runs/lora --debias none
```

- `order consistency` is substantially higher than the base model's, ideally close to
  100%.
- `letters picked` under `cyclic` is roughly flat.
- `balanced accuracy` under `none` is close to `balanced accuracy` under `cyclic`.
  When it is, the extra passes can be dropped in production.
- In the browser demo, *Shuffle option order* no longer changes the answers.
