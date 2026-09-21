#!/bin/bash
# Clone the reference repositories the benchmark reads from. Both are gitignored:
# they are other people's code and data, and are not redistributed here.
set -euo pipefail
cd "$(dirname "$0")/.."

clone() {
  if [ -d "$2" ]; then echo "$2 already present"; else
    echo "cloning $1 -> $2"; git clone --depth 1 -q "$1" "$2"
  fi
}

clone https://github.com/TheoLeeCJ/openjev.git reference-openjev
clone https://github.com/SAGAR-TAMANG/sarvam-jev.git reference-sarvam-jev

test -f reference-openjev/benchmarks/data/authored144.jsonl \
  && echo "ok: authored144 fixture is in place" \
  || { echo "authored144.jsonl missing; the upstream layout may have changed"; exit 1; }
