#include "attention.h"
#include "../kernels/kernels.cuh"
#include "../kernels/banded.cuh"
#include "../kernels/common.cuh"
#include <cmath>

// Workspace: [B,H,N,N] score tensor replaced by [B,H,N,W] band for window W.
// ws.S is reused as the band buffer (see AttentionWorkspace).
void allocate_workspace_sparse_window(AttentionWorkspace& ws,
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
    alloc(&ws.S, BHNW);
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

void attention_forward_sparse_window(cublasHandle_t handle,
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

    if (W <= 0 || (W & 1)) {
        fprintf(stderr,
            "sparse_window: window_size must be positive and even (got %d)\n", W);
        std::exit(1);
    }

    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, N, H, D, stream);

    const float scale = 1.f / std::sqrt((float)D);
    launch_qk_band(ws.Q, ws.K, ws.S, BH, N, D, W, scale, stream);

    launch_softmax_band(ws.S, BH, N, W, stream);

    launch_band_pv(ws.S, ws.V, ws.O, BH, N, D, W, stream);

    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, N, D, stream);
}
