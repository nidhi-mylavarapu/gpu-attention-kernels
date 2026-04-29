#!/usr/bin/env bash
# Run timings up through 2^13 (8192), refresh results.csv, then regenerate:
#   runtime line plot, speedup bars, peak memory plot.
#
# All C++ kernels use the same sequence lengths (fits naive’s full N×N buffer).
#
# Usage (from repo root, on a GPU node; load PyTorch + flash_attn if you want Flash rows):
#   ./benchmark_and_plot_up_to_2p13.sh
#   OUT=my.csv ./benchmark_and_plot_up_to_2p13.sh
#
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-results.csv}"
TENSOR_DIR="${TENSOR_DIR:-tensors}"
PYTHON="${PYTHON:-python3}"

# 2^7 … 2^13
SEQ_2P13="128,256,512,1024,2048,4096,8192"

if [ ! -x ./build/bench ]; then
  echo "ERROR: ./build/bench not found. Build first: cmake -B build && cmake --build build -j"
  exit 1
fi

echo "=== C++ benchmarks: naive + tiled_online + tiled_online_wmma + sparse_window (through 8192) ==="
./build/bench "$TENSOR_DIR" "$SEQ_2P13" "naive_cublas,tiled_online,tiled_online_wmma,sparse_window" > "$OUT"

echo "=== FlashAttention (Python) ==="
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "SKIP: PyTorch not available to $PYTHON — add flash_attn rows manually or:"
  echo "  module load pytorch  # etc."
else
  if ! "$PYTHON" -c "import flash_attn" 2>/dev/null; then
    echo "SKIP: flash_attn not importable — install flash-attn for official flash timings."
  else
    "$PYTHON" python/bench_flash_attn.py \
      --tensor-dir "$TENSOR_DIR" \
      --seq-lens "$SEQ_2P13" \
      --no-header >> "$OUT"
  fi
fi

echo "=== Plots → results/plots/ ==="
if ! "$PYTHON" -c "import matplotlib" 2>/dev/null; then
  echo "ERROR: matplotlib required for plotting. Example: pip install matplotlib --user"
  exit 1
fi

"$PYTHON" python/plot_runtime_vs_seqlen.py --csv "$OUT"
"$PYTHON" python/plot_speedup_bars.py --csv "$OUT"
"$PYTHON" python/plot_memory_vs_seqlen.py --csv "$OUT" --metric peak

echo
echo "Wrote consolidated CSV: $OUT"
echo "Plots:"
echo "  results/plots/runtime_vs_seq_len_dense_vs_sparse.png"
echo "  results/plots/speedup_vs_seq_len_bars.png"
echo "  results/plots/memory_vs_seq_len_peak_MB.png"
