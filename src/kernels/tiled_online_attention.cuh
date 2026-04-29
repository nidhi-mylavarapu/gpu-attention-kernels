#pragma once
#include <cuda_runtime.h>

// Tile-streamed scaled dot-product attention with online softmax (no dense S).
//
// Inputs:
//   Q, K, V:  [B*H, N, D]  row-major slabs
//   O:        [B*H, N, D]  row-major output
//   scale:    1/sqrt(D)    (caller computes; kernel does not divide)
//
// Currently compiled for D == 64 only. Pass any other D and the launch
// will exit with an error.
void launch_tiled_online_attention_bhsd(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    int B, int N, int H, int D,
    float scale,
    cudaStream_t stream = 0);
