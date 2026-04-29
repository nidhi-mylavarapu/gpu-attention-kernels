// attention_tiled_online_wmma.cu — wrapper for the WMMA (Tensor Core) variant
// of the tile-streamed online-softmax attention kernel.
//
// Outer pipeline mirrors attention_tiled_online.cu (cuBLAS Q/K/V projections,
// BSHD->BHSD transpose, attention, BHSD->BSHD transpose). The only swap is
// the attention core: launch_tiled_online_attention_wmma_bhsd instead of the
// scalar launch_tiled_online_attention_bhsd.

#include "attention.h"
#include "../kernels/kernels.cuh"
#include "../kernels/tiled_online_attention_wmma.cuh"
#include "../kernels/common.cuh"
#include <cmath>

static inline void gemm_rm(cublasHandle_t h, int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha = 1.f, float beta = 0.f) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N, A, K, &beta,
                             C, N));
}

void attention_forward_tiled_online_wmma(cublasHandle_t handle,
                                         const float* X, const float* Wq,
                                         const float* Wk, const float* Wv,
                                         float* out_BSD,
                                         AttentionWorkspace& ws,
                                         const AttentionConfig& cfg,
                                         cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    const int B  = cfg.batch;
    const int N  = cfg.seq_len;
    const int H  = cfg.n_heads;
    const int D  = cfg.d_head;
    const int Dm = cfg.d_model();
    const int BS = B * N;

    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, N, H, D, stream);

    const float scale = 1.f / std::sqrt((float)D);
    launch_tiled_online_attention_wmma_bhsd(ws.Q, ws.K, ws.V, ws.O,
                                            B, N, H, D, scale, stream);

    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, N, D, stream);
}
