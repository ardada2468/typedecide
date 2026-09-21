# Performance

The inference-speed guide is [Making the readout fast](../SPEED.md). Read this page
first; it tells you how to read that one.

## Every claim carries a label

| Label | Means |
|---|---|
| **MEASURED** | A number this project observed on hardware. The source file is named |
| **DERIVED** | Exact arithmetic from a published model config. FLOPs and bytes, never milliseconds |
| **PROJECTED** | An estimate. The assumptions are stated and the arithmetic is shown. It may be wrong |

!!! warning "What has and has not been timed"
    Nothing in the guide is a measured wall-clock benchmark **of pruning or of
    trimming**. Those two levers are DERIVED and PROJECTED only. No pruned or trimmed
    run has been committed to `bench/results/`. If you quote them, quote the label too.

## What is measured

All from `bench/results/`, on a MacBook, `onnx-community/Qwen3-0.6B-ONNX` at `q4f16`.

**Accuracy and whole-run wall clock on `authored144`** (144 rows):

| Mode | Shots | Mean family balanced accuracy | Order-unstable rows | Seconds |
|---|---:|---:|---:|---:|
| readout | 0 | 0.385 | not checked | 56.8 |
| readout | 3 | 0.419 | not checked | 189.6 |
| generate | 3 | 0.419 | not checked | 225.3 |
| readout, cyclic debias | 3 | 0.483 | 87 | 537.3 |
| readout, contextual calibration | 3 | 0.459 | 80 | 918.2 |
| readout, full permutation | 3 | 0.442 | 106 | 1102.2 |

- Readout and generation reach the same accuracy at 3 shots
  (0.41936868314577297 in both files). The readout run took 189.6 s against 225.3 s,
  about 1.19 times faster. The case for the readout is that it cannot emit a malformed
  answer, not that it is dramatically quicker.
- Cyclic debiasing took about 2.8 times as long as the single-ordering readout and
  moved balanced accuracy by +0.064.
- Cyclic (0.483) and full permutation (0.442) averaging are **not statistically
  distinguishable**: exact McNemar on their 11 discordant rows gives p = 0.23. The
  measured difference between them is run time, 537 s against 1102 s. The calibrated
  row is calibration applied to each of the 3 cyclic rotations and then averaged.
- The single-ordering result files record `order_unstable: 0` because those runs
  scored one ordering and could not observe instability. That is why the table says
  "not checked".

**Marginal cost of a suffix token**
(`logit-cost-Qwen3-0.6B-q4f16.json`, onnxruntime-node on CPU, 35-token prefix held
fixed):

| Suffix tokens | 1 | 8 | 16 | 32 | 64 | 96 |
|---|---:|---:|---:|---:|---:|---:|
| Median ms | 32.1 | 54.0 | 80.0 | 132.4 | 233.6 | 339.6 |

Fit: 28.4 ms fixed plus 3.23 ms per suffix token, so a typical 45-token criterion
projects to about 174 ms on that backend. The file's own note says this is not
comparable to the browser's WebGPU figures. The point is the shape: cost is linear in
suffix length.

## The levers, in the order the guide recommends

| Lever | Effort | Evidence |
|---|---|---|
| [Trim logits to the last position](../SPEED.md#1-the-export-returns-logits-for-every-position) | One graph edit: `trim_logits_to_last_position` | DERIVED |
| [Prune the LM head to the answer letters](../SPEED.md#2-vocabulary-pruning-slicing-the-lm-head-to-the-answer-letters) | `--prune-to N` | DERIVED. No download saving on a tied model |
| [Keep the KV cache on the GPU](../SPEED.md#4-keep-the-kv-cache-on-the-gpu) | One session option in the browser | DERIVED |
| [Quantise](../SPEED.md#3-quantisation-tiers) | `quantize(path, mode=...)` | DERIVED size. **Accuracy cost unmeasured** |
| [Shorten the per-criterion suffix](../SPEED.md#5-shorten-the-per-criterion-suffix) | Prompt work | MEASURED shape, PROJECTED size |
| [Batch the criteria](../SPEED.md#6-what-does-not-work-batched-branch-evaluation) | Not possible in the browser | MEASURED negative result |

## Do the arithmetic for your model

```bash
typedecide savings --vocab <vocab_size> --hidden <hidden_size> --keep <max options> [--tied]
```

Take `vocab_size`, `hidden_size` and `tie_word_embeddings` from your model's
`config.json`.

## The rule for any speed change

A speed change that moves accuracy is not a speed change. Score the same eval set
before and after with `--debias cyclic`, and compare `balanced_accuracy`,
`order_consistency`, then `ece` and `brier`. If you land a pruned or trimmed export
with a timing, commit its result file next to the others and move the corresponding
label in the guide from DERIVED to MEASURED.
