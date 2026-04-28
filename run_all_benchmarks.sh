#!/usr/bin/env bash
# Run all three implementations and produce a single CSV.
# Usage: ./run_all_benchmarks.sh [output.csv]
set -euo pipefail

OUT="${1:-results.csv}"
TENSOR_DIR="${TENSOR_DIR:-tensors}"

if [ ! -x ./build/bench ]; then
    echo "ERROR: ./build/bench not found. Build the C++ benchmark first."
    echo "  cmake -B build && cmake --build build -j"
    exit 1
fi

echo "=== Running C++ benchmarks (naive + simple flash) ==="
./build/bench "$TENSOR_DIR" > "$OUT"

echo "=== Running Python benchmark (official flash-attn) ==="
python3 python/bench_flash_attn.py --tensor-dir "$TENSOR_DIR" --no-header >> "$OUT"

echo
echo "Combined results in $OUT:"
echo
column -t -s, "$OUT"