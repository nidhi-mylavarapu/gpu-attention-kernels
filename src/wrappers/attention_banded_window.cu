#include "attention.h"
#include "../kernels/kernels.cuh"
#include "../kernels/banded.cuh"
#include "../kernels/common.cuh"
#include <cmath>

// Custom workspace allocator: replaces the [B,H,N,N] S tensor with a
// [B,H,N,W] band tensor. Everything else is the same as the naive workspace.
//
// Note: ws.S is reused as the band buffer. This is a slight type-pun (the
// field is named for the dense attention matrix) but lets us keep the same
// AttentionWorkspace struct. ws.S has size BH * N * W floats here, NOT
// BH * N * N.
void allocate_workspace_banded(AttentionWorkspace& ws,
                               const AttentionConfig& cfg,
                               int window_size) {
    const size_t BSD  = (size_t)cfg.batch * cfg.seq_len * cfg.d_model();
    const size_t BHSD = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len * cfg.d_head;
    const size_t BHNW = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len * window_size;

    auto alloc = [&](float** p, size_t n) {
        CUDA_CHECK(cudaMalloc(p, n * sizeof(float)));
        ws.total_bytes += n * sizeof(float);
    };
    alloc(&ws.Q_bshd, BSD);
    alloc(&ws.K_bshd, BSD);
    alloc(&ws.V_bshd, BSD);
    alloc(&ws.Q, BHSD);
    alloc(&ws.K, BHSD);
    alloc(&ws.V, BHSD);
    alloc(&ws.S, BHNW);     // band, not full attention matrix
    alloc(&ws.O, BHSD);
}

static inline void gemm_rm(cublasHandle_t h, int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha = 1.f, float beta = 0.f) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N, A, K, &beta,
                             C, N));
}

void attention_forward_banded_window(cublasHandle_t handle,
                                     const float* X,
                                     const float* Wq, const float* Wk, const float* Wv,
                                     float* out_BSD,
                                     AttentionWorkspace& ws,
                                     const AttentionConfig& cfg,
                                     cudaStream_t stream) {
    cublasSetStream(handle, stream);

    const int B  = cfg.batch;
    const int N  = cfg.seq_len;
    const int H  = cfg.n_heads;
    const int D  = cfg.d_head;
    const int Dm = cfg.d_model();
    const int W  = cfg.window_size;
    const int BS = B * N;
    const int BH = B * H;

    // Sanity: window must be > 0 and even (we use W/2 as half-width).
    if (W <= 0 || (W & 1)) {
        fprintf(stderr,
            "banded_window: window_size must be positive and even (got %d)\n", W);
        std::exit(1);
    }

    // 1. Projections (same as naive).
    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    // 2. Transpose to [B, H, S, D] (same as naive).
    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, N, H, D, stream);

    // 3. Compute band of QK^T scaled by 1/sqrt(D). OOB entries get -FLT_MAX.
    const float scale = 1.f / std::sqrt((float)D);
    launch_qk_band(ws.Q, ws.K, ws.S, BH, N, D, W, scale, stream);

    // 4. Row-wise softmax over the band.
    launch_softmax_band(ws.S, BH, N, W, stream);

    // 5. Contract the band against V to produce O.
    launch_band_pv(ws.S, ws.V, ws.O, BH, N, D, W, stream);

    // 6. Transpose back (same as naive).
    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, N, D, stream);
}
