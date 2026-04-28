// attention_sparse_window.cu
// Window attention with FLOP savings: skip out-of-window key blocks
// in both the QK^T and softmax(S)@V GEMMs. After the QK^T tile loop,
// we still run launch_window_mask to clean up the boundary entries
// inside each tile's bounding box that aren't in any query's window.
#include "attention.h"
#include "kernels.cuh"
#include "common.cuh"
#include <cmath>
#include <cfloat>
#include <algorithm>

void attention_forward_sparse_window(cublasHandle_t handle,
                                     const float* X,
                                     const float* Wq, const float* Wk, const float* Wv,
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

    const int half = cfg.window_size / 2;
    const int TQ = 512;

    // 1. Projections.
    {
        auto gemm_rm = [&](int M, int N, int K, const float* A, const float* B_, float* C) {
            const float alpha = 1.f, beta = 0.f;
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha,
                                     B_, N, A, K, &beta, C, N));
        };
        gemm_rm(BS, Dm, Dm, X, Wq, ws.Q_bshd);
        gemm_rm(BS, Dm, Dm, X, Wk, ws.K_bshd);
        gemm_rm(BS, Dm, Dm, X, Wv, ws.V_bshd);
    }

    // 2. Transpose to per-head [B, H, S, D].
    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, S, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, S, H, D, stream);

    // 3. Initialize S to -FLT_MAX so any unwritten entries are masked.
    launch_fill(ws.S, (size_t)BH * S * S, -FLT_MAX, stream);

    // 4. Sparse QK^T tile loop. Each tile writes a (tq x kw) block where
    //    kw is the union of all in-window keys for queries in the tile.
    //    This block contains some entries that are out-of-window for
    //    individual queries; we clean them up in step 4b.
    {
        const float alpha = 1.f, beta = 0.f;
        for (int q0 = 0; q0 < S; q0 += TQ) {
            int tq = std::min(TQ, S - q0);
            int k_start = std::max(0, q0 - half);
            int k_end   = std::min(S, q0 + tq + half);
            int kw      = k_end - k_start;

            const float* Q_base = ws.Q + (size_t)q0 * D;
            const float* K_base = ws.K + (size_t)k_start * D;
            float*       S_base = ws.S + (size_t)q0 * S + k_start;

            CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                kw, tq, D,
                &alpha,
                K_base, D, (long long)S * D,
                Q_base, D, (long long)S * D,
                &beta,
                S_base, S, (long long)S * S,
                BH));
        }
    }

    // 4b. Mask the boundary entries inside each tile's bounding box that
    //     fall outside individual queries' windows. Cheap O(BH*S^2) memory
    //     write, no compute.
    launch_window_mask(ws.S, BH, S, S, cfg.window_size, stream);

    // 5. Softmax.
    const float scale = 1.f / std::sqrt((float)D);
    launch_softmax_scaled(ws.S, BH * S, S, scale, stream);

    // 6. Sparse softmax(S) @ V tile loop. Out-of-window entries are 0
    //    after softmax, so although the GEMM multiplies them with V,
    //    they contribute nothing.
    {
        const float alpha = 1.f, beta = 0.f;
        for (int q0 = 0; q0 < S; q0 += TQ) {
            int tq = std::min(TQ, S - q0);
            int k_start = std::max(0, q0 - half);
            int k_end   = std::min(S, q0 + tq + half);
            int kw      = k_end - k_start;

            const float* S_base = ws.S + (size_t)q0 * S + k_start;
            const float* V_base = ws.V + (size_t)k_start * D;
            float*       O_base = ws.O + (size_t)q0 * D;

            CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, tq, kw,
                &alpha,
                V_base, D, (long long)S * D,
                S_base, S, (long long)S * S,
                &beta,
                O_base, D, (long long)S * D,
                BH));
        }
    }

    // 7. Transpose back.
    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, S, D, stream);
}