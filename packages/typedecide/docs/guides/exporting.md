# Export, prune and quantise

Shipping a tuned model to a browser is up to four steps, in this order:

1. **Export**: merge the adapter and trace the decoder to ONNX. Optionally **prune**
   the LM head during this step.
2. **Trim** the graph so it returns logits for the last position only.
3. **Quantise** to the tier the browser asks for.
4. Evaluate the artefact, because a speed change that moves accuracy is not a speed
   change.

Only step 1 has a CLI command. Trimming and quantising are library calls.

```bash
pip install "typedecide[train]" "typedecide[export]"
```

## 1. Export

=== "CLI"

    ```bash
    typedecide export --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/
    ```

=== "Python"

    ```python
    from pathlib import Path

    from typedecide.export import ExportConfig, export_onnx

    directory = export_onnx(ExportConfig(
        base_model="Qwen/Qwen3-0.6B",
        adapter=Path("runs/lora"),     # None exports the base model
        output_dir=Path("onnx"),
        opset=17,
    ))
    ```

What `export_onnx` does:

1. Loads the tokenizer and the base model in float32.
2. Merges the LoRA adapter with `merge_and_unload()`. The adapter directory must
   exist; a mismatch between adapter and base raises `ExportError` pointing you at the
   adapter's `manifest.json`.
3. If `prune_head_to` is set, prunes the head and writes `slot_map.json`.
4. Saves the merged checkpoint to `<out>/_merged`, runs Optimum's `main_export` with
   task `text-generation-with-past` at the requested opset, and writes
   `export_manifest.json`.
5. Deletes `_merged` unless `keep_merged=True` (`--keep-merged`). It is large and only
   useful for debugging.

The order is deliberate. Merging comes before pruning, because pruning a model that
still has an adapter attached would leave the adapter pointing at rows that no longer
exist. Pruning comes before the trace, because the trace freezes the output width
into the graph.

`opset` must be at least 14. 17 is the default.

### `export_manifest.json`

```json
{
  "config": {"base_model": "Qwen/Qwen3-0.6B", "adapter": "runs/lora",
             "output_dir": "onnx", "opset": 17, "prune_head_to": null,
             "keep_merged": false},
  "task": "text-generation-with-past",
  "slot_map": null,
  "pruned": false,
  "logits_are_row_indices": false,
  "can_generate_text": true,
  "versions": {"torch": "...", "transformers": "...", "optimum": "..."},
  "started": "...", "finished": "..."
}
```

## 2. Prune the LM head (optional)

A readout reads at most 20 logits. Every other row of the unembedding is computed,
copied off the accelerator and discarded on every call. Pruning slices the matrix to
the answer-letter rows, so the logits tensor goes from `[1, seq, vocab]` to
`[1, seq, K]`.

=== "CLI"

    ```bash
    typedecide export --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/ --prune-to 4
    ```

    `--prune-to N` resolves the token ids of the first `N` answer letters with
    `answer_letter_token_ids` and prints them before exporting.

=== "Python"

    ```python
    from pathlib import Path

    from typedecide.export import ExportConfig, answer_letter_token_ids, export_onnx

    keep = answer_letter_token_ids("Qwen/Qwen3-0.6B", 4)     # vocabulary ids of A, B, C, D
    export_onnx(ExportConfig(
        base_model="Qwen/Qwen3-0.6B",
        adapter=Path("runs/lora"),
        output_dir=Path("onnx-pruned"),
        prune_head_to=keep,
    ))
    ```

Set `N` to the **largest** option count of any criterion you will ever ask. A model
pruned to 4 rows cannot score a fifth option.

!!! danger "What a pruned model can no longer do. Both are permanent."
    - **It cannot generate text.** There is no output row for any token that is not a
      kept answer letter. Sampling, beam search and perplexity are gone. Keep the
      unpruned checkpoint.
    - **It cannot report the share of total vocabulary mass on the answer letters.**
      The full-vocabulary softmax denominator no longer exists. The browser demo
      computes that figure as `allowedMass`; it is the evidence that the model has
      learned the output format. A pruned model gives the softmax over the K kept rows
      and nothing else. Accuracy, ECE and Brier over the answer set are unaffected.

!!! warning "Pruning does not shrink the download of a tied model"
    When `config.tie_word_embeddings` is true, one table serves both the input
    embedding and the unembedding, and the input side still needs every row. Pruning
    then saves the matmul and the logits copy, not file size.
    `typedecide savings ... --tied` reports `weight_bytes_removed` as 0 for this
    reason. Check your model's `config.json`; do not assume.

### `slot_map.json`

After pruning, row `i` of the logits is what used to be the logit of token `keep[i]`.
Nothing in the weights records that, so the export writes it:

```json
{
  "version": 1,
  "model_id": "Qwen/Qwen3-0.6B",
  "separator": " ",
  "letters": "ABCD",
  "rows": 4,
  "original_vocab_size": 151936,
  "keep": [362, 425, 356, 422],
  "token_to_row": {"362": 0, "425": 1, "356": 2, "422": 3},
  "can_generate_text": false,
  "note": "Row index, not vocabulary id. ..."
}
```

The token ids above are illustrative. Yours come from your tokenizer.

- `keep` is in **letter order** and is never sorted. Sorting would permute the answer
  letters and produce a model that is confidently wrong.
- `token_to_row` is the inverse, keyed by decimal strings because JSON object keys are
  strings. `SlotMap.from_dict` refuses a file whose two directions disagree.
- `separator` is `" "` or `""`, whichever the [answer-slot probe](../concepts/answer-slots.md)
  selected. It is `null` when the export could not reproduce the supplied ids by
  probing. Treat `null` as a warning that the readout and the export may be scoring
  different tokens.

```python
from typedecide.export import read_slot_map

slot_map = read_slot_map("onnx-pruned")        # the directory or the file
print(slot_map.rows, slot_map.letters)
print(slot_map.row_of(slot_map.keep[1]))       # 1
print(slot_map.letter_of(1))                   # B
```

!!! warning "The browser demo does not read `slot_map.json` yet"
    `web/engine.js` discovers vocabulary ids from the tokenizer and indexes the logits
    with them. Against a K-row logits tensor those indices are out of range, and the
    page reports a non-finite score. Use pruned exports with your own runtime for now,
    and index rows `0..K-1`. See
    [Deploy to the browser demo](browser-deploy.md).

### How much would pruning save?

`typedecide savings` does the arithmetic without loading anything:

```bash
typedecide savings --vocab 151936 --hidden 1024 --keep 20 --tied
```

```text
  head_parameters                   155,582,464.00
  kept_parameters                   20,480.00
  removed_parameters                155,561,984.00
  weight_bytes_removed              0.00
  logit_bytes_per_position_before   607,744.00
  logit_bytes_per_position_after    80.00
  matmul_flops_per_position_before  311,164,928.00
  matmul_flops_per_position_after   40,960.00

These are arithmetic, not a measurement. Export a pruned model and time it.
```

The last line of that output is the honest summary. No pruned or trimmed run has been
timed and committed by this project;
[Making the readout fast](../SPEED.md) labels every such figure DERIVED or PROJECTED.

## 3. Trim logits to the last position

A stock Optimum export returns logits for **every** position. The readout reads one.
`trim_logits_to_last_position` appends a `Slice` on the sequence axis so the graph
returns `[batch, 1, vocab]`:

```python
from typedecide.export import trim_logits_to_last_position

trim_logits_to_last_position("onnx")                       # in place
trim_logits_to_last_position("onnx", output_path="onnx/model_trimmed.onnx")
```

It accepts the `.onnx` file or a directory containing `model.onnx` or
`onnx/model.onnx`. The result must pass `onnx.checker` or nothing is written. A graph
whose `logits` output is not 3-D raises `ExportError`.

Run it after the export and before quantisation. `web/engine.js` reads the final
position from the tensor's own shape, so it works against both trimmed and untrimmed
graphs.

## 4. Quantise

```python
from typedecide.export import quantize

path = quantize("onnx", mode="q4f16")      # writes model_q4f16.onnx beside the input
```

| Mode | Bits per weight | Compute | Output file | Backend |
|---|---:|---|---|---|
| `fp32` | 32 | fp32 | the input, unchanged (no-op) | none |
| `fp16` | 16 | fp16 | `model_fp16.onnx` | float16 conversion |
| `q8` | 8 | fp32 | `model_q8.onnx` | ONNX Runtime dynamic quantisation of `MatMul` weights |
| `q4` | 4.5 | fp32 | `model_q4.onnx` | `MatMul4BitsQuantizer`, block size 32, symmetric |
| `q4f16` | 4.5 | fp16 | `model_q4f16.onnx` | the same, then float16 conversion |

The 4.5 bits are 4 bits of weight plus one fp16 scale per block of 32
(`4 + 16 / block_size`). `block_size` must be a positive multiple of 16. The output
names follow the transformers.js convention so `web/models.json` can point at them.
The float16 conversion needs `onnxconverter-common` or ONNX Runtime's transformers
tools; the error says so if neither is present.

!!! warning "Quantise after pruning, and measure the accuracy cost"
    Quantising first spends bits on rows that are about to be deleted. And this
    project has **not measured** the accuracy cost of any tier, so neither should you
    assume it. Score the same eval set at each tier and compare `balanced_accuracy`
    and `order_consistency`. A tier that keeps accuracy but drops order consistency
    has degraded the model into a position-guesser.

## Pruning an in-memory model yourself

`prune_lm_head` works on any causal LM with `get_output_embeddings()` or `.lm_head`:

```python
from transformers import AutoModelForCausalLM

from typedecide.export import (
    answer_letter_token_ids, build_slot_map, prune_lm_head, write_slot_map,
)

model_id = "Qwen/Qwen3-0.6B"
keep = answer_letter_token_ids(model_id, 4)
model = AutoModelForCausalLM.from_pretrained(model_id)

prune_lm_head(model, keep)                      # logits are now [batch, seq, 4]
write_slot_map(build_slot_map(keep, model_id=model_id, separator=" "), "pruned/")
model.save_pretrained("pruned/")
```

It unties tied embeddings before slicing and verifies the input embedding is
untouched before returning. The trap it avoids is described in the
[ONNX export runbook](../runbooks/onnx-export-fails.md).
