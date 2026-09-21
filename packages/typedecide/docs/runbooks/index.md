# Runbooks

Operational, step-by-step procedures organised by symptom. Every runbook has the same
four sections:

- **Symptoms**: what you see, including the exact message where there is one.
- **Diagnosis**: commands to run to find out which cause you have.
- **Fix**: what to change, per cause.
- **Verify**: how to confirm it is fixed.

## Data

| Symptom | Runbook |
|---|---|
| `typedecide validate` exits 1 or 2 on an export from another system | [Validation fails on a customer export](validation-fails.md) |
| `A split ... would leave N class(es) with zero examples in eval` | [Split refuses, class missing from eval](split-refuses.md) |

## Training

| Symptom | Runbook |
|---|---|
| `No answer-letter continuation keeps the prompt boundary stable` | [PromptError on a new tokenizer](prompt-error-tokenizer.md) |
| Loss is flat, `nan`, or stuck near `ln(n_options)` | [Training loss not decreasing](loss-not-decreasing.md) |
| The adapter scores below the base model | [Tuned model scores worse than base](tuned-worse-than-base.md) |
| Low `order_consistency`, one letter dominates `letters picked` | [Model answers by position](answers-by-position.md) |

## Shipping

| Symptom | Runbook |
|---|---|
| `prompt parity: FAILED` in CI | [Prompt parity check fails in CI](prompt-parity-ci.md) |
| `The ONNX export failed`, or a reloaded pruned model has no head | [ONNX export fails or reload has no head](onnx-export-fails.md) |
| The page runs on WASM, or reports a non-finite score | [Browser falls back to WASM or non-finite scores](browser-wasm-nonfinite.md) |
| You need to publish a version | [Releasing a new version](releasing.md) |

## Before you open an issue

Collect these. They answer most first questions:

```bash
typedecide -v validate <your file> --json-out report.json   # debug log + report
python -c "import typedecide; print(typedecide.__version__)"
pip list | grep -i -E "typedecide|torch|transformers|peft|optimum|onnx"
```

and, for anything after training, the run's `manifest.json`. Remove customer text from
anything you attach.
