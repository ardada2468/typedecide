# Deploy to the browser demo

The repository's `web/` directory is a static page with no build step and no backend.
It loads a model with transformers.js, prefills the state once, and scores each
criterion against that cache. This documentation site serves a copy of it at
[`/demo/`](../../demo/).

## Run the page locally

```bash
cd web
python3 -m http.server 8090
# open http://localhost:8090
```

The stock model (`onnx-community/Qwen3-0.6B-ONNX`, about 450 MB according to
`web/models.json`) downloads once and is cached by the browser. WebGPU is used when
`navigator.gpu` exists, WebAssembly otherwise, and the page says which it got.

## How the page chooses a model

`web/models.json` lists models. **`web/app.js` loads the first entry** (`models[0]`);
there is no model picker. Each entry looks like this:

```json
{
  "id": "qwen3-0.6b",
  "name": "Qwen3 0.6B",
  "repo": "onnx-community/Qwen3-0.6B-ONNX",
  "dtype": { "webgpu": "q4f16", "wasm": "q4" },
  "size": "~450 MB",
  "role": "base",
  "style": "completion"
}
```

`repo` and `dtype` are passed to transformers.js:
`AutoTokenizer.from_pretrained(repo)` and
`AutoModelForCausalLM.from_pretrained(repo, { device, dtype })`. The engine tries
`webgpu` with `dtype.webgpu` first, then `wasm` with `dtype.wasm`.

## Put your fine-tuned model behind it

1. **Export without pruning**, then trim and quantise to both tiers the page asks for:

    ```python
    from pathlib import Path

    from typedecide.export import (
        ExportConfig, export_onnx, quantize, trim_logits_to_last_position,
    )

    out = export_onnx(ExportConfig(
        base_model="Qwen/Qwen3-0.6B", adapter=Path("runs/lora"),
        output_dir=Path("export/triage"),
    ))
    trim_logits_to_last_position(out)
    quantize(out, mode="q4f16")     # WebGPU tier
    quantize(out, mode="q4")        # WASM tier
    ```

2. **Lay the files out the way transformers.js expects**: the tokenizer and config
   files at the top level of the model repository, and the ONNX files in an `onnx/`
   subfolder named by tier (`onnx/model_q4f16.onnx`, `onnx/model_q4.onnx`). Optimum
   writes `model.onnx` at the top level of the output directory, so move the quantised
   files into `onnx/`.

3. **Host the repository where the browser can fetch it.** The route this project has
   exercised is a Hugging Face Hub model repository, which is how the stock entry
   works. Upload the folder and use its `owner/name` as `repo`.

4. **Make it the first entry in `web/models.json`.**

5. **Reload and check the status line.** It names the backend and tier it loaded. Run
   the example ticket with *Shuffle option order* on and off.

!!! warning "Serving the model from `web/models/` is not wired up"
    `web/models.json` contains a second entry with `"repo": "models/qwen3-0.6b-triage-onnx"`
    and `"local": true`, and `web/models/` is gitignored for that purpose. But
    `web/app.js` never reads the `local` flag, and `web/engine.js` does not configure
    transformers.js for local model paths. Expect to change the page's code before a
    same-origin model folder loads. Treat that entry as a placeholder.

!!! danger "Do not deploy a pruned export to this page"
    `web/engine.js` indexes the returned logits with **vocabulary ids** found by
    `answerSlots`, and computes `allowedMass` over the full vocabulary. A pruned
    export returns K rows, so those indices fall outside the tensor and the page
    throws *"An answer letter scored non-finite; the vocabulary does not match this
    prompt."* The page does not read `slot_map.json`. See the
    [runbook](../runbooks/browser-wasm-nonfinite.md).

## Keep the prompt identical

The page builds its prompt with `web/prompt.js`; your fine-tune used
`typedecide/prompt.py`. They must be byte-identical. If you edit the instruction or
the option rendering in one, change the other and retrain. See
[Prompt parity](../concepts/prompt-parity.md).

The rubric you type into the page (questions, option descriptions, keys) is data. It
does not need to match anything in Python, but the closer the questions and option
descriptions are to the ones you trained on, the more the fine-tune applies.

## Hosting notes

**Cross-origin isolation.** `web/_headers` sets

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

for hosts that honour a `_headers` file (Netlify and Cloudflare Pages do).
Multi-threaded WebAssembly needs those headers, because it depends on
`SharedArrayBuffer`.

!!! note "GitHub Pages cannot set these headers"
    The copy of the demo served from this documentation site at `/demo/` is hosted on
    GitHub Pages, which ignores `_headers` and offers no way to set response headers.
    WebGPU is unaffected. The WebAssembly fallback still runs, without cross-origin
    isolation and therefore without threads, so it is slower there than on a host
    that sends the headers.

**Privacy.** There is no backend. The evidence text a visitor types is tokenized and
scored in their tab. The only network traffic is the page's static files, the
transformers.js library from jsDelivr, fonts from Google Fonts, and the model files
from wherever `repo` points.

**Caching.** Model files are cached by the browser after the first load. Changing
`repo` or the tier triggers a new download.

## What the page demonstrates, carefully

From the repository README, measured on the reference machine: given a well-engineered
prompt, the generation lane answers all ten example criteria correctly in about the
same order of time as the readout (1.5 s against 1.15 s). The honest advantages are
time to first decision (about 310 ms for the first readout), structurally valid
output, a probability per option, and determinism. Do not present the demo as a large
speed-up.
