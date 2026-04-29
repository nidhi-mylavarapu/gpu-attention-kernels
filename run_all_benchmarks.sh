#!/usr/bin/env bash
# Run C++ bench (naive, tiled_online, sparse_window) and append Python flash-attn CSV.
# Usage: ./run_all_benchmarks.sh [output.csv]
set -euo pipefail

OUT="${1:-results.csv}"
TENSOR_DIR="${TENSOR_DIR:-tensors}"

if [ ! -x ./build/bench ]; then
    echo "ERROR: ./build/bench not found. Build the C++ benchmark first."
    echo "  cmake -B build && cmake --build build -j"
    exit 1
fi

echo "=== Running C++ bench (naive, tiled_online, sparse_window) ==="
echo "    SEQ_LENS=${SEQ_LENS:-(default)}  IMPLS=${IMPLS:-(default)}"
./build/bench "$TENSOR_DIR" > "$OUT"

echo "=== Running Python benchmark (official flash-attn) ==="
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not available to $PYTHON."
    echo "  On NERSC GPU nodes, load the PyTorch module (not the generic python module), e.g.:"
    echo "    module load pytorch"
    echo "  Then install flash-attn once: pip install --user flash-attn"
    echo "  See README.md section \"Run the Python flash-attn benchmark\"."
    exit 1
fi
"$PYTHON" python/bench_flash_attn.py --tensor-dir "$TENSOR_DIR" --no-header >> "$OUT"

echo
echo "Combined results in $OUT:"
echo
column -t -s, "$OUT"