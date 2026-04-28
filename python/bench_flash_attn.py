#!/usr/bin/env python3
"""
Benchmark the official flash-attn library on the same input data as the
C++ benchmark.

Loads X, Wq, Wk, Wv from the tensor dump produced by `./build/bench`,
performs the same projection+attention forward pass, and prints CSV
matching the format of the C++ benchmark output so they can be concatenated.

Important: flash-attn requires FP16 or BF16. We do projections in FP32 to
match the C++ baseline arithmetic, then cast Q/K/V to FP16 before calling
flash-attn. This is the canonical way real flash-attn workloads run.
"""
import argparse
import os
import sys
import math
import numpy as np
import torch

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("ERROR: flash_attn not installed. Try: pip install --user flash-attn",
          file=sys.stderr)
    sys.exit(1)


# Must match main.cu defaults
BATCH   = 1
N_HEADS = 8
D_HEAD  = 64
D_MODEL = N_HEADS * D_HEAD
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 3
ITERS  = 10


def load_float32(path: str, expected_count: int) -> torch.Tensor:
    """Load a raw float32 file and return a 1D CUDA tensor."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} floats, got {arr.size}")
    return torch.from_numpy(arr).cuda()


def attention_forward_flash_attn(X, Wq, Wk, Wv, n_heads, d_head):
    """
    Same forward pass as the C++ benchmarks, but using flash-attn for the
    attention core. Projections in FP32, attention in FP16.

    Shapes:
      X:  [B, S, D_model]  (FP32)
      Wq, Wk, Wv: [D_model, D_model]  (FP32)
    Returns:
      O:  [B, S, D_model]  (FP32)
    """
    B, S, Dm = X.shape
    H, D = n_heads, d_head

    # FP32 projections (matches what cuBLAS does in the C++ baseline).
    # Note: cuBLAS on Ampere+ may use TF32 tensor cores by default for sgemm,
    # so this "FP32" matmul is actually TF32 unless we set CUBLAS_PEDANTIC_MATH.
    # PyTorch matmul on Ampere+ also uses TF32 by default. The behavior
    # therefore matches the C++ baseline closely.
    Q = X @ Wq                      # [B, S, Dm]
    K = X @ Wk
    V = X @ Wv

    # Reshape to [B, S, H, D] (flash-attn's expected layout).
    Q = Q.view(B, S, H, D)
    K = K.view(B, S, H, D)
    V = V.view(B, S, H, D)

    # flash-attn requires FP16 or BF16. FP16 to match common deployment.
    Q16 = Q.to(torch.float16)
    K16 = K.to(torch.float16)
    V16 = V.to(torch.float16)

    # Non-causal, no dropout, default scale (which is 1/sqrt(D)).
    O16 = flash_attn_func(Q16, K16, V16, dropout_p=0.0, causal=False)
    # Output: [B, S, H, D], FP16. Cast back to FP32 and flatten heads.
    O = O16.to(torch.float32).reshape(B, S, Dm)
    return O


def benchmark_one(seq_len: int, tensor_dir: str):
    """Benchmark one sequence length using files already on disk."""
    Dm = D_MODEL
    BSD = BATCH * seq_len * Dm
    WSZ = Dm * Dm

    base = f"{tensor_dir}/n{seq_len}_b{BATCH}_h{N_HEADS}_d{D_HEAD}"
    fX  = f"{base}_X.bin"
    fWq = f"{base}_Wq.bin"
    fWk = f"{base}_Wk.bin"
    fWv = f"{base}_Wv.bin"
    for f in (fX, fWq, fWk, fWv):
        if not os.path.exists(f):
            raise RuntimeError(
                f"Missing {f}. Run the C++ benchmark first to generate tensors.")

    X  = load_float32(fX,  BSD).view(BATCH, seq_len, Dm)
    Wq = load_float32(fWq, WSZ).view(Dm, Dm)
    Wk = load_float32(fWk, WSZ).view(Dm, Dm)
    Wv = load_float32(fWv, WSZ).view(Dm, Dm)

    # Reset memory tracking before this benchmark so peak_MB is meaningful.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(WARMUP):
        _ = attention_forward_flash_attn(X, Wq, Wk, Wv, N_HEADS, D_HEAD)
    torch.cuda.synchronize()

    # Timed runs using CUDA events (same approach as GpuTimer in C++).
    times_ms = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = attention_forward_flash_attn(X, Wq, Wk, Wv, N_HEADS, D_HEAD)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    peak_bytes = torch.cuda.max_memory_allocated()

    # We don't easily separate "workspace" from "everything else" in PyTorch,
    # so we report peak_allocated as both workspace and peak. peak_MB in the
    # C++ output uses cudaMemGetInfo which includes the CUDA context overhead;
    # the PyTorch numbers exclude that, so they will appear smaller. That's
    # actually a fairer measurement of true workspace.
    return {
        "impl":        "flash_attn_official",
        "seq_len":     seq_len,
        "mean_ms":     sum(times_ms) / len(times_ms),
        "min_ms":      min(times_ms),
        "max_ms":      max(times_ms),
        # PyTorch tracks allocations not memory pool reservations; this is
        # the cleanest equivalent of "workspace size" for this benchmark.
        "workspace_MB": peak_bytes / (1024 * 1024),
        "peak_MB":      peak_bytes / (1024 * 1024),
        "max_abs_err":  "skip",  # no CPU reference for FP16 results
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-dir", default="tensors",
                    help="Directory containing tensor dumps from the C++ benchmark")
    ap.add_argument("--no-header", action="store_true",
                    help="Skip CSV header (useful when concatenating with C++ output)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    if not args.no_header:
        print("impl,seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err")

    for S in SEQ_LENS:
        try:
            r = benchmark_one(S, args.tensor_dir)
            print(f"{r['impl']},{r['seq_len']},{r['mean_ms']:.3f},"
                  f"{r['min_ms']:.3f},{r['max_ms']:.3f},"
                  f"{r['workspace_MB']:.1f},{r['peak_MB']:.1f},"
                  f"{r['max_abs_err']}")
        except Exception as e:
            print(f"flash_attn_official,{S},NaN,NaN,NaN,NaN,NaN,error: {e}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()