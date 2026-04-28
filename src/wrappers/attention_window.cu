// attention_window.cu
#include "attention.h"
#include "../kernels/kernels.cuh"
#include "../kernels/common.cuh"
#include <cmath>

// Row-major C[M,N] = A[M,K] * B[K,N]  via cuBLAS.
static inline void gemm_rm(cublasHandle_t h, int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha = 1.f, float beta = 0.f) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N, A, K, &beta,
                             C, N));
}

void attention_forward_window(cublasHandle_t handle,
                              const float* X, const float* Wq,
                              const float* Wk, const float* Wv,
                              float* out_BSD,
                              AttentionWorkspace& ws,
                              const AttentionConfig& cfg,
                              cudaStream_t stream) {
    cublasSetStream(handle, stream);

    const int B  = cfg.batch;
    const int S  = cfg.seq_len;
    const int H  = cfg.n_heads;
    const int D  = cfg.d_head;
    const int Dm = cfg.d_model();
    const int BS = B * S;
    const int BH = B * H;

    // 1. Projections.
    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    // 2. Reshape transpose.
    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, S, H, D, stream);

    // 3. Scores: S_mat = Q @ K^T.
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            S, S, D,
            &alpha,
            ws.K, D, (long long)S * D,
            ws.Q, D, (long long)S * D,
            &beta,
            ws.S, S, (long long)S * S,
            BH));
    }

    // 3b. NEW: apply window mask before softmax.
    if (cfg.window_size > 0) {
        launch_window_mask(ws.S, BH, S, S, cfg.window_size, stream);
    }

    // 4. Scale + softmax row-wise.
    const float scale = 1.f / std::sqrt((float)D);
    launch_softmax_scaled(ws.S, BH * S, S, scale, stream);

    // 5. Context: O = softmax(S) @ V.
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, S, S,
            &alpha,
            ws.V, D, (long long)S * D,
            ws.S, S, (long long)S * S,
            &beta,
            ws.O, D, (long long)S * D,
            BH));
    }

    // 6. Transpose back.
    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, S, D, stream);
}