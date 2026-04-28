#pragma once
#include <cuda_runtime.h>

// =============================================================================
// Banded window attention kernels.
//
// Band layout: S_band has shape [B*H, N, W]. For query row i, band column c
// corresponds to absolute key index k = i - half + c, where half = W / 2.
// Out-of-range keys (k < 0 or k >= N) are masked with -FLT_MAX so softmax
// outputs 0 for them.
// =============================================================================

// S_band[b,h,i,c] = scale * dot(Q[b,h,i,:], K[b,h,k,:]) if k in [0, N), else -FLT_MAX
// where k = i - half + c.
// Shapes: Q, K [B*H, N, D]   S_band [B*H, N, W]
void launch_qk_band(const float* Q, const float* K, float* S_band,
                    int BH, int N, int D, int W, float scale,
                    cudaStream_t stream = 0);

// Row-wise numerically-stable softmax on a band of width W.
// One block per row of the band (BH * N rows total).
void launch_softmax_band(float* S_band, int BH, int N, int W,
                         cudaStream_t stream = 0);

// O[b,h,i,d] = sum_c S_band[b,h,i,c] * V[b,h,k,d]   (k = i - half + c, skipped if OOB)
// Shapes: V [B*H, N, D]   S_band [B*H, N, W]   O [B*H, N, D]
void launch_band_pv(const float* S_band, const float* V, float* O,
                    int BH, int N, int D, int W,
                    cudaStream_t stream = 0);
