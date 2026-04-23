# gpu-attention-kernels

A CUDA/C++ implementation of naive scaled dot-product attention (forward pass), built as a performance baseline for comparing against tiled, fused, and FlashAttention-style optimizations.

This is a **systems/performance project, not an ML project** — no backward pass, no training, no model quality. The only goal is to measure runtime and memory usage of a realistic attention workload and to serve as an honest baseline for future optimization work.

## What it does

Implements the standard forward pass:

```
Q = X W_q,  K = X W_k,  V = X W_v
S = softmax(Q K^T / sqrt(d_head))
O = S V
```

with:

- **FP32 throughout** (correctness-first baseline; FP16/BF16/tensor-cores are a later optimization axis)
- **cuBLAS** for projections and the two batched matmuls (`Q·Kᵀ` and `P·V`)
- **Custom kernels** only for row-wise numerically-stable softmax and the `[B,S,H,D] ↔ [B,H,S,D]` transposes
- **Full attention** (non-causal), no masking
- **Materialized `S` matrix** — the whole point of the baseline; this is exactly what FlashAttention avoids

## Project layout

```
├── CMakeLists.txt
├── common.cuh          # CUDA error macros, GPU event timer, memory helpers
├── kernels.cuh         # kernel declarations
├── kernels.cu          # softmax + transpose kernels
├── attention.h         # public API + workspace struct
├── attention.cu        # naive forward pass orchestration
├── reference.cpp       # CPU reference for correctness
└── main.cu             # benchmark driver
```

The `attention_forward_naive(...)` entry point takes `X, W_q, W_k, W_v` and writes to an output buffer. Future optimized implementations (tiled, fused, FlashAttention) keep the same signature so `main.cu` can swap them in without touching the harness.

## Build

Requirements: CUDA 11+, CMake 3.18+, a C++17 compiler.

```bash
cmake -B build
cmake --build build -j
```

`CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 80 86 89` (A100, RTX 30xx, RTX 40xx). Edit for your GPU; e.g. `90` for H100.

If CMake can't find cuBLAS, pass `-DCUDAToolkit_ROOT=/path/to/cuda` (the directory containing `bin/nvcc`).

## Run

```bash
./build/bench
```

Output is CSV to stdout:

```
seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err
128,0.173,0.171,0.179,2.2,470.1,0.000000
256,0.223,0.222,0.230,5.5,474.1,0.000000
512,0.355,0.354,0.358,15.0,486.1,0.000000
1024,0.659,0.656,0.663,46.0,518.1,skip
2048,1.936,1.933,1.939,156.0,632.1,skip
4096,5.845,5.835,5.900,568.0,1052.1,skip
8192,17.328,17.313,17.339,2160.0,2660.1,skip
```

- `workspace_MB`: sum of all intermediate device allocations (Q, K, V, S, O, plus pre-transpose buffers)
- `peak_MB`: actual used GPU memory as reported by `cudaMemGetInfo`, including cuBLAS workspace and CUDA context overhead
- `max_abs_err`: max element-wise deviation from the CPU reference (`skip` for seq > 512 because the CPU version is O(N²·D) and gets slow)

## Configuration

Defaults in `main.cu`:

```cpp
batch   = 1
n_heads = 8
d_head  = 64
seq_lens = {128, 256, 512, 1024, 2048, 4096, 8192}
warmup = 3
iters  = 10
```

These are chosen so the largest `S` tensor (~2 GB at `seq=8192`) fits comfortably on a 16GB+ GPU. If you have less memory, drop `seq=8192` or reduce `n_heads`. If you have more and want to stress-test, add `16384` — but note the `S` tensor is then ~17 GB.

## Expected behavior

Attention is O(N²) in both time and memory once model dimensions are fixed. You should see:

- **Doubling ratios approach 4×** as N grows — the projections (linear in N) become negligible next to the `Q·Kᵀ`, softmax, and `P·V` kernels
- **Workspace memory dominated by `S`**: `B · H · N² · 4 bytes`. At defaults, that's 16 MB at N=1024 and 2 GB at N=8192
- **Sub-quadratic at short N** due to launch overhead — below ~N=256, you're kernel-launch-bound, not compute-bound
- **`max_abs_err` on the order of 1e-6 to 1e-5** vs the CPU reference, with `--use_fast_math` on

If you see `max_abs_err` much larger than that, it's almost always a layout bug in one of the cuBLAS calls. Print a few output elements side-by-side with the reference.

## Profiling

### nsys (timeline + kernel summary)

```bash
nsys profile --stats=true -o report ./build/bench
```

**On HPC clusters** (Lustre home directories on Cray/HPE systems), `nsys` report assembly may fail with errno 524. Write the report to scratch:

```bash
nsys profile --stats=true -o $SCRATCH/bench_report ./build/bench
```

The CUDA Kernel Statistics section will show, per iteration at large N:

- 3× `sgemm` calls for projections (small)
- 1× batched `sgemm` for `Q·Kᵀ` (large)
- `softmax_scaled_kernel` (bandwidth-bound, substantial)
- 1× batched `sgemm` for `P·V` (large)
- 2× `transpose_*` kernels (modest)

The ratio `softmax_time / (QKT_time + PV_time)` is the most interesting number here — softmax does far less arithmetic than the GEMMs but touches the same multi-gigabyte `S` tensor, so on bandwidth alone it takes a meaningful fraction of total time. This is exactly the traffic a fused/online-softmax implementation eliminates.

### ncu (per-kernel deep dive)

```bash
ncu --set full -k "softmax_scaled_kernel" ./build/bench
```

For the softmax kernel, check **achieved DRAM throughput vs peak** — it should approach peak on long rows and fall far short on short rows (launch overhead).

For the cuBLAS GEMMs, throughput vs peak tells you whether that kernel is memory-bound or compute-bound. The `P·V` matmul at long N is especially interesting because `S` is enormous and mostly-filled with non-trivial values.

## Correctness

`reference.cpp` provides a straightforward CPU implementation using triple-nested loops. `main.cu` invokes it for `seq_len ≤ 512` and reports `max_abs_err` against the GPU output. At the default config and N(0, 0.02) initialization, expected error is ≤ 1e-5 in FP32. If you disable `--use_fast_math` in `CMakeLists.txt`, expect error to drop by an order of magnitude (at the cost of slower `expf`).

## Roadmap

This repo is deliberately a baseline. Planned comparison points, in roughly increasing order of effort:

1. **Tiled shared-memory transpose** — fix the uncoalesced writes in the naive transpose kernels
2. **Fused scale + softmax** with vectorized loads (`float4`) — remove one pass over the `S` tensor
3. **FP16/BF16 with tensor cores** via `cublasGemmEx` — expect 4–8× on the GEMMs
4. **Online softmax / tiled attention** (FlashAttention-style) — the big one. Should collapse both runtime and peak memory at long N, and is where the baseline really earns its keep as a comparison point.

Each should drop in behind the same `attention_forward_*` signature so `main.cu` remains unchanged.

## License

MIT.


***
# GPU Attention Kernels Benchmark

This project implements and benchmarks different GPU-based attention mechanisms, including naive, tiled, and fused variants. The goal is to study performance trade-offs and memory behavior for large sequence lengths on modern GPUs.

---

## Project Structure

```
gpu-attention-kernels/
├── benchmarks/        # Benchmark driver (main entry point)
├── src/
│   ├── kernels/      # Low-level CUDA kernels (softmax, transpose, etc.)
│   ├── wrappers/     # Full attention implementations
│   └── reference.cpp # CPU reference implementation (for correctness)
├── data/             # Pre-generated input tensors (.npy)
├── python/           # Data generation scripts
├── build/            # Build directory (generated)
└── CMakeLists.txt
```

---

## Setup

### 1. Build

```bash
mkdir -p build
cd build
cmake ..
make -j
```

---

## Data Generation

We use fixed input data to ensure reproducible benchmarking across runs.

Generate data:

```bash
cd python
python3 generate_data.py
```

This creates:

```
data/
  X_128.npy
  Wq_128.npy
  Wk_128.npy
  Wv_128.npy
  ...
```

---

## Running on Perlmutter (GPU REQUIRED)


### Allocate a GPU node:

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account=<acc>
```

### Verify GPU:

```bash
nvidia-smi
```

---

## Running Benchmarks

From the `build/` directory:

```bash
./bench naive
./bench tiled
./bench fused
```

---

## Output Format

```
kernel=naive
seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err
128,0.160,...
256,0.215,...
...
```

### Metrics

- **mean_ms / min_ms / max_ms** → runtime statistics  
- **workspace_MB** → memory used by intermediate buffers  
- **peak_MB** → peak GPU memory usage  
- **max_abs_err** → correctness vs CPU reference  
  - `skip` = correctness not checked (large inputs)

---

## Correctness Checking

For small sequence lengths (≤ 512), we compare:

```
GPU implementation vs CPU reference
```

using max absolute error.

Large inputs skip correctness due to high CPU cost.

---

## Adding New Implementations

### Step 1: Add wrapper

Create a new file:

```
src/wrappers/attention_<name>.cu
```

Implement:

```cpp
void attention_forward_<name>(...);
```

---

### Step 2: Register in `attention.h`

```cpp
void attention_forward_<name>(...);
```

---

### Step 3: Add to benchmark

Update kernel selection in `benchmark.cu`:

```cpp
if (kernel == "<name>") {
    attention_fn = attention_forward_<name>;
}
```

---

### Step 4: Add to CMake

```cmake
src/wrappers/attention_<name>.cu
```

---

### Step 5: Rebuild

```bash
cd build
rm -rf *
cmake ..
make -j
```

---

## Benchmarking Workflow

```
generate_data.py → data/ → benchmark.cu → GPU kernels → results
```

---

