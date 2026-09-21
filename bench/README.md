# bench

Two scripts, each backing one claim in the top-level README.

```bash
npm install
```

**`shared_check.mjs` — shared-prefix KV reuse is exact.** Scores the same criteria
twice, once with a full prompt each time and once with the state prefilled once and
reused, then reports the maximum probability drift between them. Expect `0.000e+0`.

```bash
MODEL=onnx-community/Qwen3-0.6B-ONNX DTYPE=fp32 node shared_check.mjs
```

**`rotation_probe.mjs` — does the model read the evidence, or pick a letter?** Eight
unambiguous decisions, scored under every rotation of the option list, reporting
accuracy, how often the answer survives reordering, and the distribution of letters
picked. A model with a positional prior is obvious in the last two columns.

```bash
MODEL=onnx-community/Qwen3-0.6B-ONNX DTYPE=q4f16 node rotation_probe.mjs
MODEL=onnx-community/gemma-3-270m-it-ONNX DTYPE=fp32 node rotation_probe.mjs
```

Eight items is a signal, not a benchmark. For a real measurement use
`train/evaluate.py`, which runs the same metric over the full fixture.
