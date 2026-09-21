#!/bin/bash
# Does model size fix this, and at what download? One row per Qwen3 size, same
# prompt, same fixture, readout only -- the permutation modes are 3-6x the work and
# the larger models are slow enough on CPU that they are better run on a GPU box.
set -u
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  echo "=== $model ==="
  node authored144.mjs --model "onnx-community/${model}-ONNX" --mode readout --shots 3 \
    2>&1 | grep -vE '^\s+[0-9]+/[0-9]+$'
  echo
done
