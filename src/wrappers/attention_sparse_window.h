#pragma once
#include <cuda_runtime.h>

// =============================================================================
// Sparse (sliding) window attention: scores live in a band buffer [B*H, N, W],
// not a full N×N matrix. Implemented via launch_qk_band / softmax / PV
// (see banded.cuh).
//
// Band layout: S_band has shape [B*H, N, W]. For query row i, band column c
// corresponds to absolute key index k = i - half + c, where half = W / 2.
// Out-of-range keys (k < 0 or k >= N) are masked with -FLT_MAX so softmax
// outputs 0 for them.
// =============================================================================

void launch_qk_band(const float* Q, const float* K, float* S_band,
                    int BH, int N, int D, int W, float scale,
                    cudaStream_t stream = 0);

void launch_softmax_band(float* S_band, int BH, int N, int W,
                         cudaStream_t stream = 0);

void launch_band_pv(const float* S_band, const float* V, float* O,
                    int BH, int N, int D, int W,
                    cudaStream_t stream = 0);
