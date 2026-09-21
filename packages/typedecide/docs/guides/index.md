# How-to guides

Task-oriented pages. Each assumes you have read
[Getting started](../getting-started/index.md) and want to do one thing properly.

| Guide | Use it when |
|---|---|
| [Bring your own data](bring-your-own-data.md) | You have a CSV, JSON, Parquet or Hugging Face dataset in your own column names |
| [Read a validation report](validation-report.md) | `typedecide validate` printed findings and you need to know which ones matter |
| [Configure training](training.md) | You are writing `train.yaml` |
| [Evaluate like a sceptic](evaluating.md) | You have a number and want to know whether to believe it |
| [Export, prune and quantise](exporting.md) | You are producing an ONNX artefact |
| [Deploy to the browser demo](browser-deploy.md) | You want your model behind the page in `web/` |
| [Handle errors](error-handling.md) | You are calling the library from your own code or CI |
| [Reproducibility and manifests](reproducibility.md) | You need to trace a result back to its inputs |

If something has already gone wrong, the [runbooks](../runbooks/index.md) are
organised by symptom.
