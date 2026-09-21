# Reproducibility and manifests

A result you cannot trace back to its input is not a result. Three stages write a
record of what produced their output.

| Stage | Where | Written by |
|---|---|---|
| Training | `<output_dir>/manifest.json`, and `TrainResult.manifest` | `finetune` |
| Evaluation | `EvalResult.manifest`, and `manifest` in the `--json-out` file | `evaluate` |
| Export | `<output_dir>/export_manifest.json`, plus `slot_map.json` when pruned | `export_onnx` |

## The training manifest

```json
{
  "manifest_version": 1,
  "base_model": "Qwen/Qwen3-0.6B",
  "config": {
    "base_model": "Qwen/Qwen3-0.6B", "output_dir": "runs/lora",
    "epochs": 2.0, "batch_size": 4, "grad_accum": 4, "effective_batch_size": 16,
    "learning_rate": 0.0001, "lora_rank": 16, "lora_alpha": 32,
    "lora_alpha_was_explicit": false, "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    "max_length": 1024, "seed": 0, "randomise_option_order": true,
    "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "logging_steps": 25
  },
  "seed": 0,
  "dataset": {
    "train": {"fingerprint": "<sha256>", "rows": 96,
              "criteria": ["queue", "refund_requested"], "groups": 48},
    "eval":  {"fingerprint": "<sha256>", "rows": 24,
              "criteria": ["queue", "refund_requested"], "groups": 12}
  },
  "answer_slots": {"queue": {"A": 0, "B": 0, "C": 0}},
  "option_order_randomisation": "per-epoch",
  "timing": {"started_at": "...", "finished_at": "...", "wall_seconds": 0.0},
  "versions": {"python": "...", "platform": "...", "torch": "...",
               "transformers": "...", "peft": "...", "accelerate": "...",
               "numpy": "...", "typedecide": "..."},
  "argv": ["typedecide", "train", "--config", "train.yaml"],
  "losses": {"train": 0.0, "eval": 0.0},
  "steps": 0,
  "trainable_parameters": {"trainable": 0, "total": 0}
}
```

The numeric values above are placeholders showing the shape. Points worth knowing:

- `config` is **resolved**. `lora_alpha` is the number peft received, never `null`, and
  the fixed policy constants are included so the manifest still describes the run if
  a later release changes them.
- `dataset.*.fingerprint` is a SHA-256 over the decisions sorted by id. It does not
  depend on row order. It does depend on content, including option order.
- `answer_slots` maps each criterion key to `{letter: token id}` as resolved by the
  preflight. If the same key resolved differently for different option sets, a warning
  was logged and the last one is recorded.
- `argv` is the command line of the process that called `finetune`, with anything
  that looks like a credential replaced by `[redacted]`: the value after a flag whose
  name contains `token`, `secret`, `password`, `api-key`, `credential` or `auth`, and
  any argument shaped like a Hugging Face, OpenAI-style or GitHub token.
- `versions` reports a missing library as `"not installed"`, never by omission.
  `typedecide` is `"source"` for an uninstalled checkout.
- The manifest records the training rows that were actually used. Unlabelled eval
  rows are dropped before the eval fingerprint is taken.

## The fingerprint

```python
from typedecide import fingerprint, load_decisions

print(fingerprint(load_decisions("data/eval.jsonl")))
```

Use it to prove two results were computed on the same data. The evaluation manifest
records the fingerprint of the **labelled rows it scored**, so it equals the value
above only when every row in the file is labelled.

## The evaluation manifest

| Key | Meaning |
|---|---|
| `model`, `adapter`, `device`, `batch_size` | What was scored and how (absent for an injected scorer) |
| `debias`, `orderings_scored` | The mode, and the largest number of orderings scored for any decision |
| `forward_passes` | Scored prompts, including content-free passes under `calibrated` |
| `decisions`, `unlabelled_skipped` | Rows scored and rows skipped |
| `fingerprint` | Hash of the scored rows |
| `chance` | Mean of 1/n over the scored rows |
| `seed`, `python`, `platform`, `started`, `seconds` | Run context |

## What determinism you can expect

- **Data operations are exactly reproducible.** `group_split` sorts group keys before
  shuffling with `random.Random(seed)`, so the result does not depend on load order.
  `randomise_option_order` seeds each row from `(seed, decision.id)`, so a row gets
  the same ordering in any dataset.
- **The readout does not sample.** Scoring the same prompt with the same weights on the
  same hardware and library versions gives the same distribution. `TorchReadout`
  seeds torch anyway.
- **Training is seeded, not bit-reproducible across hardware.** `seed_everything` seeds
  Python, NumPy, torch (and CUDA) and transformers before the model is built, and the
  per-epoch permutations derive from the same seed. GPU kernels, library versions and
  dtype can still change low-order bits. The manifest records enough to explain a
  difference; it cannot prevent one.

## A checklist for a result you intend to quote

- [ ] The eval set came from `group_split`, with `group_id` set where rows share a state.
- [ ] `validate` reports no errors, and you have read the warnings.
- [ ] Base and tuned models were scored with the same file, `--debias` and seed, and the
      two evaluation fingerprints match.
- [ ] `order_consistency` is reported beside the accuracy.
- [ ] `manifest.json`, both evaluation JSON files and (if exported)
      `export_manifest.json` are stored together.
- [ ] Any number you quote names its fixture, model, quantisation tier and debias mode.
