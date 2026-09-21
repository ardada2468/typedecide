# Install

## The base package

```bash
pip install typedecide
```

This gives you loading, validation, splitting, option-order augmentation, the metrics,
the prompt builder and the CLI. Its only dependencies are `numpy` and `pyyaml`.
`import typedecide` does not import torch.

## Extras

Heavy dependencies are opt-in. Install the extra for the stage you need:

| Extra | Installs | Needed for |
|---|---|---|
| `train` | `torch>=2.4`, `transformers>=4.51`, `peft>=0.13`, `accelerate>=0.34`, `datasets>=3.0` | `typedecide train`, `typedecide evaluate`, loading a Hugging Face dataset id |
| `export` | `optimum[exporters]>=1.23`, `onnx>=1.16`, `onnxruntime>=1.20` | `trim_logits_to_last_position`, `quantize`, and the ONNX half of `typedecide export` |
| `parquet` | `pyarrow>=15.0` | Reading `.parquet` / `.pq` files |
| `dev` | `pytest`, `pytest-cov`, `mypy`, `ruff` | Working on the library itself |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building this site. CI installs the pinned `docs/requirements.txt` instead |
| `all` | `train` + `export` + `parquet` | Every runtime extra. Not `dev` or `docs` |

```bash
pip install "typedecide[train]"
pip install "typedecide[export]"
pip install "typedecide[parquet]"
pip install "typedecide[all]"
```

!!! warning "Exporting needs both `train` and `export`"
    `export_onnx` loads the base model with transformers and merges the adapter with
    peft before handing it to Optimum, so it needs the `train` extra as well as
    `export`. The error message says so if either is missing:
    `pip install 'typedecide[export]' 'typedecide[train]'`.

!!! tip "Quote the brackets"
    In zsh (the macOS default), `pip install typedecide[train]` fails with
    `no matches found`. Quote it: `pip install "typedecide[train]"`.

## From a source checkout

The library lives in `packages/typedecide` of the repository:

```bash
git clone https://github.com/ardada2468/typedecide
cd typedecide
pip install -e "packages/typedecide[train,export]"
```

## Check the install

```bash
typedecide --help
typedecide savings --vocab 151936 --hidden 1024 --keep 20 --tied
```

`savings` is pure arithmetic and needs no extras, so it is a quick way to confirm the
console script is on your `PATH`. In Python:

```python
import typedecide

print(typedecide.__version__)
```

## What happens when an extra is missing

Nothing fails at import time. The function that needs the dependency raises one of the
library's own exceptions, with the install command in the message:

| You called | Missing | Raises |
|---|---|---|
| `load_decisions("x.parquet")` | `pyarrow` | `DataError`: "... Fix: pip install typedecide[parquet]" |
| `load_decisions("owner/name")` | `datasets` | `DataError`: "... Fix: pip install typedecide[train]" |
| `finetune(...)` | `torch`, `transformers` or `peft` | `TrainingError` naming each missing package |
| `evaluate(...)` | `torch`, `transformers` or `peft` | `ConfigError` naming the package |
| `export_onnx(...)`, `quantize(...)` | any exporter dependency | `ExportError` |

The CLI prints these as `error: ...` and exits with status 2. See
[Handle errors](../guides/error-handling.md).

## Hardware

- **Validate, split, augment, metrics:** any machine.
- **Train:** `finetune` loads the base model in bfloat16 when CUDA is available and
  float32 otherwise. CPU training works and is slow.
- **Evaluate:** the device defaults to `cuda` when available and `cpu` otherwise. Any
  other device, including Apple `mps`, has to be requested explicitly with
  `--device mps` or `device="mps"`.
