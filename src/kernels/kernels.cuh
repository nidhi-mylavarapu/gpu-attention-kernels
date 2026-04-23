// kernels.cuh
#pragma once

// S <- S / scale, then softmax(S) row-wise. S shape: [rows, row_len].
void launch_softmax_scaled(float* S, int rows, int row_len, float scale,
                           cudaStream_t stream = 0);

// [B, S, H, D] -> [B, H, S, D]
void launch_transpose_bshd_to_bhsd(const float* src, float* dst,
                                   int B, int S, int H, int D,
                                   cudaStream_t stream = 0);

// [B, H, S, D] -> [B, S, H, D]
void launch_transpose_bhsd_to_bshd(const float* src, float* dst,
                                   int B, int H, int S, int D,
                                   cudaStream_t stream = 0);