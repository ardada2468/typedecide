# Browser demo falls back to WASM, or returns non-finite scores

## Symptoms

- The status line reports `wasm` where you expected `webgpu`, and each decision takes
  seconds.
- The page shows *"Could not start the model on WebGPU or WASM. ..."*
- A run stops with *"An answer letter scored non-finite; the vocabulary does not match
  this prompt."*
- A run stops with *"No answer-letter continuation keeps the prompt boundary stable
  for this tokenizer."* or *"The state prefix is not a token-level prefix of this
  prompt."*

## Diagnosis

### Falling back to WASM

`web/engine.js` tries WebGPU only when `navigator.gpu` exists, and falls back to WASM
if WebGPU is absent **or if loading on WebGPU throws**. Open the browser's developer
console and run:

```js
navigator.gpu ? await navigator.gpu.requestAdapter() : 'navigator.gpu is undefined'
```

| Result | Meaning |
|---|---|
| `'navigator.gpu is undefined'` | The browser has no WebGPU, it is disabled, or the page is not a secure context. WebGPU needs `https://` or `localhost` |
| `null` | WebGPU exists and no adapter is available: blocklisted GPU or driver, a virtual machine, a remote desktop session, or hardware acceleration turned off |
| A `GPUAdapter` object | WebGPU works. The model failed to load on it, so look at the console and network tabs for the real error |

Common load failures with a working adapter: the WebGPU tier file is missing from the
model repository (`dtype.webgpu` in `web/models.json` is `q4f16`, so the page requests
an `onnx/model_q4f16.onnx`), or the device ran out of memory.

**Is WASM itself slow?** Check `crossOriginIsolated` in the console. `false` means no
`SharedArrayBuffer`, so ONNX Runtime's WebAssembly backend runs without threads. The
headers in `web/_headers` provide isolation on hosts that honour that file. GitHub
Pages does not, so the copy of the demo at `/demo/` on this site is never
cross-origin isolated.

### Non-finite scores

`readSlots` takes the logits of the answer-letter **vocabulary ids** at the last
position and throws if any is not a finite number. Causes, most likely first:

| Cause | How to confirm |
|---|---|
| The export was **pruned**. Logits are `[1, seq, K]`, and vocabulary ids index past the end | The model folder contains `slot_map.json`, or `export_manifest.json` has `"pruned": true` |
| Tokenizer and model come from different checkpoints | Compare `vocab_size` in the repository's `config.json` with the tokenizer's vocabulary |
| The quantised graph overflows in fp16 on this GPU | The same model works on WASM, or at another tier |
| A corrupt or partial download | The network tab shows a failed or truncated model file |

## Fix

**WebGPU unavailable**

- Use a browser with WebGPU enabled, and make sure hardware acceleration is on.
- Serve over `https://` or from `localhost`. A page opened from `file://` or a plain
  `http://` LAN address has no `navigator.gpu`.
- Update GPU drivers. On a VM or remote desktop there may be no fix; accept WASM.

**WebGPU available, model will not load on it**

- Publish the tier the page asks for:

    ```python
    from typedecide.export import quantize

    quantize("export/triage", mode="q4f16")    # WebGPU tier
    quantize("export/triage", mode="q4")       # WASM tier
    ```

    and place the results under `onnx/` in the model repository. Or change `dtype` in
    `web/models.json` to a tier you do publish.

**WASM is slow**

- Host the page somewhere that sends `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. `web/_headers` already says so for
  Netlify and Cloudflare Pages. With `require-corp`, every cross-origin resource must
  allow being embedded, so re-test the model download after enabling it.
- Shorten the per-criterion suffix. Cost is linear in suffix tokens; see
  [Making the readout fast](../SPEED.md#5-shorten-the-per-criterion-suffix).

**Non-finite scores**

- **Pruned export:** the page cannot use it. `web/engine.js` does not read
  `slot_map.json`. Re-export **without** `--prune-to` for the demo, and keep pruned
  exports for a runtime that indexes rows `0..K-1`.
- **Mismatched tokenizer:** export the tokenizer and the model together. `export_onnx`
  writes both from the same base.
- **fp16 overflow:** try `dtype: { "webgpu": "fp16" }` or a `q4` tier to isolate it,
  and compare accuracy across tiers before settling.
- **Corrupt download:** clear the site's cache storage and reload.

**Prefix or answer-slot errors** mean the tokenizer does not behave the way the prompt
needs. They are the browser-side equivalents of
[PromptError on a new tokenizer](prompt-error-tokenizer.md).

## Verify

- The status line names `webgpu` and the tier you intended.
- A run on the example ticket completes, and every criterion shows finite
  probabilities that sum to 1.
- The reported share of vocabulary mass on the answer letters is high. It is the
  evidence that the model has learned the output format.
- With *Shuffle option order* ticked, a fine-tuned model's answers do not change. If
  they do, continue with [Model answers by position](answers-by-position.md).
