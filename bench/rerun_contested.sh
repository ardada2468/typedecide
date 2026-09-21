#!/bin/bash
# Re-run the two modes whose ranking is in question, this time saving per-row
# predictions so the difference can be tested rather than eyeballed.
set -u
for mode in readout-cyclic readout-full; do
  echo "=== $mode ==="
  node authored144.mjs --mode "$mode" --shots 3 2>&1 | grep -vE '^\s+[0-9]+/[0-9]+$'
done
node compare.mjs results/authored144-Qwen3-0.6B-ONNX-q4f16-readout-cyclic-shots3.predictions.jsonl \
                 results/authored144-Qwen3-0.6B-ONNX-q4f16-readout-full-shots3.predictions.jsonl
