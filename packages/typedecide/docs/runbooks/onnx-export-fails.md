# ONNX export fails, or a pruned model reloads with no head

## Symptoms

One of:

- `error: 'optimum.exporters.onnx' is needed to export ONNX but is not installed.`
- `error: No adapter at runs/lora. Point adapter at the directory finetune wrote ...`
- `error: Merging runs/lora into <base> failed: ...`
- `error: The ONNX export failed: ... If the model has a pruned head, check that
  config.vocab_size still describes the *input* embedding ...`
- `error: Token id N at position P is outside the model's vocabulary of V ...`
- `error: opset 13 is below 14 ...`
- You pruned a model in your own script, saved it with `save_pretrained`, and on
  reload transformers warns that `lm_head.weight` was newly initialised, or the
  logits are `[batch, seq, vocab]` again, or the model answers at chance.

## Diagnosis

**Is the environment complete?** Exporting needs both extras.

```bash
python -c "import torch, transformers, peft, optimum.exporters.onnx, onnx, onnxruntime; print('ok')"
```

**Is the adapter what you think it is?**

```bash
ls runs/lora                      # expect adapter_config.json, adapter weights, manifest.json
python -c "import json; print(json.load(open('runs/lora/manifest.json'))['base_model'])"
```

`--base` must be that base model.

**Does the unpruned export work?** This separates an exporter problem from a pruning
problem, and `--keep-merged` leaves the intermediate checkpoint for inspection.

```bash
typedecide -v export --base <base> --adapter runs/lora --out /tmp/export-plain --keep-merged
```

**For a pruned checkpoint, inspect the staged model:**

```bash
typedecide -v export --base <base> --adapter runs/lora --out /tmp/export-pruned --prune-to 4 --keep-merged
python - <<'PY'
import json
config = json.load(open("/tmp/export-pruned/_merged/config.json"))
print("tie_word_embeddings:", config.get("tie_word_embeddings"))   # must be False
print("vocab_size:", config.get("vocab_size"))                     # must be the ORIGINAL size
PY
cat /tmp/export-pruned/slot_map.json
```

### The `_tied_weights_keys` trap

Many small models set `tie_word_embeddings = true`: `lm_head.weight` **is**
`model.embed_tokens.weight`, the same tensor object. Three things go wrong if you
slice the head naively:

| Naive step | Consequence |
|---|---|
| Slice `lm_head.weight` in place | The *input* embedding now has K rows. Every prompt token id above K indexes out of bounds or reads the wrong row. The model still runs, and accuracy quietly collapses |
| Leave `config.tie_word_embeddings = True` | On reload, transformers re-ties the K-row head onto the input embedding and destroys it |
| Leave `model._tied_weights_keys` set | `save_pretrained` treats `lm_head.weight` as a duplicate of the input embedding and **omits it from the saved weights**. The reloaded model has no trained head. Nothing fails until inference |

A fourth, related mistake: lowering `config.vocab_size` to K. The input side still
needs a row per token id, and Optimum then builds out-of-range dummy inputs during the
trace, which is the export failure quoted above.

## Fix

| Cause | Fix |
|---|---|
| Missing dependency | `pip install "typedecide[train]" "typedecide[export]"` |
| Adapter path wrong | Pass the directory `finetune` wrote, the one holding `adapter_config.json` |
| Adapter and base mismatch | Export against `manifest["base_model"]` |
| Opset too low | Use `--opset 17` (the default). 14 is the minimum |
| Kept token id outside the vocabulary | The ids came from a different tokenizer. Recompute them with `answer_letter_token_ids(<the same base>, N)`, or use `--prune-to N` and let the CLI do it |
| Duplicate or negative kept id | Same fix. A duplicate would make two letters share a row forever |
| Hand-rolled pruning | Replace it with `prune_lm_head`, below |
| You changed `config.vocab_size` | Do not. Leave it at the original value |
| Optimum does not support the architecture | Confirm with the unpruned export. If that fails too, it is an Optimum limitation for that model family, not a typedecide bug |

**Use `prune_lm_head`, never a manual slice.** It rebinds the head to a fresh K-row
tensor, sets `config.tie_word_embeddings = False`, clears `_tied_weights_keys`, leaves
`config.vocab_size` alone, and then verifies that the input embedding still has its
original row count and values and no longer shares storage with the head. If any check
fails, it raises `ExportError` and does not return a quietly broken model.

```python
from transformers import AutoModelForCausalLM

from typedecide.export import (
    answer_letter_token_ids, build_slot_map, prune_lm_head, write_slot_map,
)

model_id = "Qwen/Qwen3-0.6B"
keep = answer_letter_token_ids(model_id, 4)
model = AutoModelForCausalLM.from_pretrained(model_id)

prune_lm_head(model, keep)
model.save_pretrained("pruned/")
write_slot_map(build_slot_map(keep, model_id=model_id, separator=" "), "pruned/")
```

If you have an adapter, merge it **before** pruning (`export_onnx` does this for you).
Pruning first leaves the adapter pointing at rows that no longer exist.

## Verify

```python
import torch
from transformers import AutoModelForCausalLM

from typedecide.export import read_slot_map

slot_map = read_slot_map("pruned/")
model = AutoModelForCausalLM.from_pretrained("pruned/")

head = model.get_output_embeddings().weight
embed = model.get_input_embeddings().weight
assert head.shape[0] == slot_map.rows, head.shape           # K rows out
assert embed.shape[0] == slot_map.original_vocab_size       # full table in
assert head.data_ptr() != embed.data_ptr()                  # untied
assert model.config.tie_word_embeddings is False

with torch.inference_mode():
    logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
assert logits.shape[-1] == slot_map.rows
print("pruned checkpoint reloads with its head:", tuple(logits.shape))
```

- The script passes, and loading printed no warning about newly initialised weights.
- For an ONNX export, `export_manifest.json` has `"pruned": true` and
  `"logits_are_row_indices": true`, and `slot_map.json` sits beside the model with a
  non-null `separator`.
- Score the artefact through a [`ScoreFn`](../guides/evaluating.md#score-something-that-is-not-a-local-pytorch-model)
  and confirm `balanced_accuracy` and `order_consistency` match the PyTorch model's.
