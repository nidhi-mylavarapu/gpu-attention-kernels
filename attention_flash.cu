#include "attention_flash.h"
#include "kernels.cuh"
#include "common.cuh"
#include <cmath>
#include <cfloat>

// =============================================================================
// Fused tiled attention with online softmax.
//
// Grid:  (num_q_tiles, B*H)     — one block per (batch_head, Q tile)
// Block: (Br) threads           — one thread per Q row in the tile
//
// Each thread owns:
//   - one row of the score tile S_ij (length Bc)
//   - one row of the output accumulator O_i (length D)
//   - scalar running max m and running normalizer ℓ
//
// Shared memory holds the current Q tile and the current K, V tiles.
// =============================================================================

template <int Br, int Bc, int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,   // [B*H, N, D]
    const float* __restrict__ K,   // [B*H, N, D]
    const float* __restrict__ V,   // [B*H, N, D]
    float* __restrict__ O,         // [B*H, N, D]
    int N,
    float scale)
{
    const int bh = blockIdx.y;             // batch_head index
    const int q_tile = blockIdx.x;         // which tile of Q rows
    const int tid = threadIdx.x;           // which row within the tile (0..Br-1)

    const int q_row_global = q_tile * Br + tid;
    const bool row_valid = q_row_global < N;

    // Pointers to this batch_head's slabs
    const float* Qp = Q + (size_t)bh * N * D;
    const float* Kp = K + (size_t)bh * N * D;
    const float* Vp = V + (size_t)bh * N * D;
    float*       Op = O + (size_t)bh * N * D;

    // Shared memory tiles.
    __shared__ float sQ[Br][D];
    __shared__ float sK[Bc][D];
    __shared__ float sV[Bc][D];

    // Load this thread's row of Q into shared memory.
    // Each thread reads its own Br row, stripping across D.
    if (row_valid) {
        for (int d = 0; d < D; ++d)
            sQ[tid][d] = Qp[(size_t)q_row_global * D + d];
    } else {
        for (int d = 0; d < D; ++d)
            sQ[tid][d] = 0.f;
    }

    // Per-thread (per-row) running statistics, in registers.
    float m_i = -FLT_MAX;          // running max
    float l_i = 0.f;               // running normalizer
    float O_reg[D];                // running unnormalized output
    #pragma unroll
    for (int d = 0; d < D; ++d) O_reg[d] = 0.f;

    // Per-thread scratch for the score row of the current tile.
    float S_row[Bc];

    __syncthreads();

    // Sweep through K/V tiles.
    const int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_base = kv_tile * Bc;

        // Cooperative load of K and V tiles into shared memory.
        // Br threads load Bc * D elements each for K and V.
        // With Br = Bc = 64, D = 64, each thread loads D elements of K and V.
        // Generalize: each thread loads ceil(Bc*D / Br) elements per tile.
        constexpr int LOADS_PER_THREAD = (Bc * D + Br - 1) / Br;
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            int idx = tid + i * Br;
            if (idx < Bc * D) {
                int row = idx / D;
                int col = idx % D;
                int g_row = kv_base + row;
                if (g_row < N) {
                    sK[row][col] = Kp[(size_t)g_row * D + col];
                    sV[row][col] = Vp[(size_t)g_row * D + col];
                } else {
                    sK[row][col] = 0.f;
                    sV[row][col] = 0.f;
                }
            }
        }
        __syncthreads();

        // Compute this thread's row of S = Q_row · K_tile^T, scaled.
        // Out-of-range columns get -inf so they contribute nothing to softmax.
        float row_max_local = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < Bc; ++j) {
            int k_row_global = kv_base + j;
            if (k_row_global < N && row_valid) {
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < D; ++d)
                    dot += sQ[tid][d] * sK[j][d];
                dot *= scale;
                S_row[j] = dot;
                if (dot > row_max_local) row_max_local = dot;
            } else {
                S_row[j] = -FLT_MAX;
            }
        }

        // Online softmax update.
        float m_new = fmaxf(m_i, row_max_local);

        // alpha rescales prior accumulators to the new max.
        // Guard against the all-(-inf) case on the very first tile.
        float alpha = (m_i == -FLT_MAX) ? 0.f : __expf(m_i - m_new);

        // Compute P_ij = exp(S_row - m_new) and its row sum.
        float row_sum = 0.f;
        float P_row[Bc];
        #pragma unroll
        for (int j = 0; j < Bc; ++j) {
            float p = (S_row[j] == -FLT_MAX) ? 0.f : __expf(S_row[j] - m_new);
            P_row[j] = p;
            row_sum += p;
        }

        // Update running normalizer and output.
        l_i = alpha * l_i + row_sum;

        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float pv = 0.f;
            #pragma unroll
            for (int j = 0; j < Bc; ++j)
                pv += P_row[j] * sV[j][d];
            O_reg[d] = alpha * O_reg[d] + pv;
        }

        m_i = m_new;

        // Don't overwrite sK/sV in next iteration before everyone finishes.
        __syncthreads();
    }

    // Final normalization and write-out.
    if (row_valid) {
        float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
            Op[(size_t)q_row_global * D + d] = O_reg[d] * inv_l;
    }
}

// =============================================================================
// Workspace allocation: like the naive version but skips the [B, H, S, S] S.
// =============================================================================
void allocate_workspace_flash(AttentionWorkspace& ws, const AttentionConfig& cfg) {
    const size_t BSD  = (size_t)cfg.batch * cfg.seq_len * cfg.d_model();
    const size_t BHSD = (size_t)cfg.batch * cfg.n_heads * cfg.seq_len * cfg.d_head;

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
    alloc(&ws.O, BHSD);
    ws.S = nullptr;     // never allocated
}

// =============================================================================
// Same row-major helper as in attention.cu.
// =============================================================================
static inline void gemm_rm(cublasHandle_t h, int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha = 1.f, float beta = 0.f) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N, A, K, &beta,
                             C, N));
}

void attention_forward_flash(cublasHandle_t handle,
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
    const int BH = B * H;

    // Same as naive: projections via cuBLAS, then head-major transpose.
    gemm_rm(handle, BS, Dm, Dm, X, Wq, ws.Q_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wk, ws.K_bshd);
    gemm_rm(handle, BS, Dm, Dm, X, Wv, ws.V_bshd);

    launch_transpose_bshd_to_bhsd(ws.Q_bshd, ws.Q, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.K_bshd, ws.K, B, N, H, D, stream);
    launch_transpose_bshd_to_bhsd(ws.V_bshd, ws.V, B, N, H, D, stream);

    // Fused attention. Tile sizes are compile-time constants.
    constexpr int Br = 64;
    constexpr int Bc = 64;

    if (D != 64) {
        fprintf(stderr,
                "flash kernel currently compiled only for D=64 (got %d). "
                "Add a template instantiation to support other head dims.\n", D);
        std::exit(1);
    }

    const float scale = 1.f / std::sqrt((float)D);
    const int num_q_tiles = (N + Br - 1) / Br;

    dim3 grid(num_q_tiles, BH);
    dim3 block(Br);

    flash_attention_kernel<Br, Bc, 64><<<grid, block, 0, stream>>>(
        ws.Q, ws.K, ws.V, ws.O, N, scale);

    // Same head-de-transpose as naive.
    launch_transpose_bhsd_to_bshd(ws.O, out_BSD, B, H, N, D, stream);
}
