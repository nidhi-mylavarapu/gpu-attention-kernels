// attention.cu
#include "attention.h"
#include "../kernels/kernels.cuh"
#include "../kernels/common.cuh"
#include <cmath>

void allocate_workspace(AttentionWorkspace& ws, const AttentionConfig& cfg) {
    const size_t BSD = (size_t)cfg.batch * cfg.seq_len * cfg.d_model();
    const size_t BHSD = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len * cfg.d_head;
    const size_t BHSS = (size_t)cfg.batch * cfg.n_heads
                        * cfg.seq_len * cfg.seq_len;

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
    alloc(&ws.S, BHSS);
    alloc(&ws.O, BHSD);
}

void free_workspace(AttentionWorkspace& ws) {
    cudaFree(ws.Q_bshd); cudaFree(ws.K_bshd); cudaFree(ws.V_bshd);
    cudaFree(ws.Q);      cudaFree(ws.K);      cudaFree(ws.V);
    cudaFree(ws.S);      cudaFree(ws.O);
    ws = AttentionWorkspace{};
}

// Row-major C[M,N] = A[M,K] * B[K,N]  via cuBLAS.
// Rule: call cuBLAS with (opB, opA, N, M, K, B, ldB=N, A, ldA=K, C, ldC=N).
static inline void gemm_rm(cublasHandle_t h, int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha = 1.f, float beta = 0.f) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N, A, K, &beta,
                             C, N));
}

void attention_forward_naive(cublasHandle_t handle,
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
    const int BS = B * S;                    // flat "rows" for projection
    const int BH = B * H;                    // batched matmul count

    // 1. Projections:  [BS, Dm] = [BS, Dm] @ [Dm, Dm]
    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    // 2. Reshape [B, S, Dm] as [B, S, H, D], transpose to [B, H, S, D].
    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, S, H, D, stream);

    // 3. Scores: S_mat = Q @ K^T, shape [S, S] per (b, h).
    //    Strided batched. A = Q (no trans), B = K (trans), C = S_mat.
    //    Row-major C[S, S] = A[S, D] * B^T[D, S]  where B stored [S, D].
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,          // op on K (in row-major terms)
            CUBLAS_OP_N,          // op on Q
            S, S, D,
            &alpha,
            ws.K, D, (long long)S * D,    // B = K, ldb = D (row stride), stride
            ws.Q, D, (long long)S * D,    // A = Q, lda = D
            &beta,
            ws.S, S, (long long)S * S,    // C, ldc = S, stride
            BH));
    }

    // 4. Scale + softmax row-wise. Total rows = B * H * S.
    const float scale = 1.f / std::sqrt((float)D);
    launch_softmax_scaled(ws.S, BH * S, S, scale, stream);

    // 5. Context: O = softmax(S) @ V.  [S, D] = [S, S] @ [S, D]
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

    // 6. Transpose [B, H, S, D] -> [B, S, H, D] into output.
    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, S, D, stream);
}