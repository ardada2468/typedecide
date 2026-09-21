#!/bin/bash
# Every row of the ablation table, on the full 144-row fixture.
set -u
for spec in "readout 0" "readout 3" "generate 3" "readout-cyclic 3" "readout-calibrated 3" "readout-full 3"; do
  set -- $spec
  echo "=== mode=$1 shots=$2 ==="
  node authored144.mjs --mode "$1" --shots "$2" 2>&1 | grep -vE '^\s+[0-9]+/[0-9]+$'
  echo
done
