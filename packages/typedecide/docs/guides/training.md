# Configure training

A fine-tuning job is fully described by one frozen
[`TrainConfig`](../reference/api/training/config.md) plus the dataset. Everything that
changes the weights lives in it.

## A complete config

```yaml title="train.yaml"
base_model: Qwen/Qwen3-0.6B     # required: a Hugging Face id or a local path
output_dir: runs/triage-lora    # required: adapter, tokenizer and manifest.json land here

epochs: 2.0
batch_size: 4
grad_accum: 4                   # effective batch = batch_size * grad_accum = 16
learning_rate: 1.0e-4

lora_rank: 16
lora_alpha: null                # null or absent means 2 * lora_rank
lora_dropout: 0.05

max_length: 1024                # tokens, including the one answer token
seed: 0
randomise_option_order: true    # re-permute options at the start of every epoch
```

```bash
typedecide train --config train.yaml --train data/train.jsonl --eval data/eval.jsonl
```

The file may be YAML (`.yaml`, `.yml`) or JSON (`.json`). Any other suffix is tried as
JSON and then as YAML.

!!! warning "Relative `output_dir` is relative to the config file"
    `TrainConfig.from_file` resolves a relative `output_dir` against the **config
    file's own directory**, not your working directory, so a config can move with its
    run. If `train.yaml` is in `configs/`, then `output_dir: runs/lora` means
    `configs/runs/lora`. A `TrainConfig` built directly in Python is not rewritten.

## Every field

| Field | Type | Default | Constraint | What it does |
|---|---|---|---|---|
| `base_model` | `str` | required | non-empty | Hugging Face model id or local path, loaded with `AutoModelForCausalLM` |
| `output_dir` | `Path` | required | non-empty | Receives the LoRA adapter, the tokenizer and `manifest.json`. A `str` is accepted and converted |
| `epochs` | `float` | `2.0` | `> 0` | Passed to `num_train_epochs`. Fractions are allowed |
| `batch_size` | `int` | `4` | `>= 1` | Per-device batch size for training and for the eval loss |
| `grad_accum` | `int` | `4` | `>= 1` | Gradient accumulation steps |
| `learning_rate` | `float` | `1e-4` | `> 0` | Peak learning rate |
| `lora_rank` | `int` | `16` | `>= 1` | LoRA `r` |
| `lora_alpha` | `int` or `None` | `None` | `>= 1` when set | `None` means twice the rank. Read the resolved value from `effective_lora_alpha` |
| `lora_dropout` | `float` | `0.05` | `0.0 <= x < 1.0` | LoRA dropout |
| `max_length` | `int` | `1024` | `>= 16` | Maximum example length in tokens. Over-length states are truncated from the left |
| `seed` | `int` | `0` | `>= 0` | Seeds Python, NumPy, torch and transformers, the data order, and the per-epoch option permutations |
| `randomise_option_order` | `bool` | `True` | | Re-permute every training decision's options at the start of each epoch |

Violating a constraint raises [`ConfigError`](../reference/api/errors.md) naming the
field and the value. A config file with a key that is not in this table is rejected
with the list of permitted keys, so a typo such as `learning-rate` cannot silently fall
back to the default. In files, booleans may also be written as `yes`/`no`, `on`/`off`,
`1`/`0`. A boolean where a number is expected is rejected, as are `nan` and `inf`, and
a whole-number field with a fractional part (`batch_size: 2.7`) is an error, not
rounded down.

Two derived properties:

```python
from pathlib import Path

from typedecide.training import TrainConfig

config = TrainConfig(base_model="Qwen/Qwen3-0.6B", output_dir=Path("runs/lora"))
print(config.effective_lora_alpha)    # 32
print(config.effective_batch_size)    # 16
```

## What is policy, not configuration

These are module constants in `typedecide.training.config`. They cannot be set from a
config file, and they are written into every manifest so a run stays reproducible if a
later release changes them.

| Constant | Value |
|---|---|
| `LORA_TARGET_MODULES` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `LR_SCHEDULER_TYPE` | `cosine` |
| `WARMUP_RATIO` | `0.03` |
| `LOGGING_STEPS` | `25` |

Other fixed behaviour of `finetune`:

- LoRA `bias="none"`, `task_type="CAUSAL_LM"`.
- The base model loads in bfloat16 when CUDA is available, float32 otherwise, and
  `bf16` training is enabled under the same condition.
- A checkpoint is saved each epoch and only the latest is kept
  (`save_total_limit=1`). With an eval set, eval loss is computed each epoch.
- No experiment tracker is enabled (`report_to=[]`).
- If the tokenizer has no pad token, the EOS token is used. Padding is masked out of
  attention and carries `-100` labels, so the choice cannot reach the loss.

!!! note "Architectures that name their projections differently"
    If LoRA cannot attach to the base model, `finetune` raises `TrainingError` listing
    the target modules. The target list is not configurable in this release.

## From Python

```python
from pathlib import Path

from typedecide import load_decisions
from typedecide.training import TrainConfig, finetune

train = load_decisions("data/train.jsonl")
held = load_decisions("data/eval.jsonl")


def progress(event: dict) -> None:
    if event.get("event") == "log" and "loss" in event:
        print(f"step {event['step']}: loss {event['loss']:.4f}")


config = TrainConfig.from_file(Path("train.yaml"))
result = finetune(train, config, eval_set=held, progress=progress)

print(result.adapter_dir, result.train_loss, result.eval_loss)
print(result.manifest["answer_slots"])
```

`progress` receives dicts whose `event` key is one of:

| `event` | Other keys |
|---|---|
| `start` | `train_rows`, `eval_rows` |
| `preflight` | `criteria` (sorted criterion keys whose answer slots resolved) |
| `epoch` | `epoch` |
| `log` | `step`, plus whatever the Hugging Face trainer logged (`loss`, `learning_rate`, `eval_loss`, ...) |
| `done` | `adapter_dir`, `train_loss`, `eval_loss` |

An exception raised by your callback is logged and swallowed. A broken progress bar
does not kill a long run.

## Choosing values

The defaults are a starting point for a 0.6B model and a few thousand decisions. The
project has not published a hyperparameter sweep, so the advice here is procedural:

- **Change one thing at a time and keep the manifests.** Each run records its config
  and dataset fingerprint; see [Reproducibility](reproducibility.md).
- **Leave `randomise_option_order` on.** Turning it off is the one setting that
  reliably produces a model that [answers by position](../runbooks/answers-by-position.md).
- **Size `max_length` to your states.** Run `validate` first; a `long_state` warning
  means rows will be truncated. Truncation is from the left of the state and is safe
  for the answer slot, but evidence is still being dropped.
- **Judge a run by evaluation, not by loss.** Score the base model and the tuned model
  the same way with `--debias cyclic`, and compare `balanced_accuracy` and
  `order_consistency`.

If the loss is not moving at all, use the
[runbook](../runbooks/loss-not-decreasing.md).
