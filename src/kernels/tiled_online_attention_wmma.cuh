#pragma once
#include <cuda_runtime.h>

// Tile-streamed scaled dot-product attention with online softmax,
// using Tensor Cores (nvcuda::wmma) for both Q*K^T and P*V.
//
// Math is identical to launch_tiled_online_attention_bhsd. The difference is
// that Q/K/V are cast FP32 -> FP16 on the fly into shared memory, the two
// matmuls run on Tensor Cores with FP32 accumulators, and online softmax is
// done in FP32 in shared memory.
//
// Inputs:
//   Q, K, V:  [B*H, N, D]  row-major slabs (FP32 in global memory)
//   O:        [B*H, N, D]  row-major output (FP32)
//   scale:    1/sqrt(D)
//
// Currently compiled for D == 64 only.
void launch_tiled_online_attention_wmma_bhsd(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int B, int N, int H, int D,
    float scale,
    cudaStream_t stream = 0);
