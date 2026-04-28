#include "banded.cuh"
#include "common.cuh"
#include <cfloat>

// =============================================================================
// QK·band: compute the band of scores.
//
// Grid:  (N, BH)        — one block per (batch_head, query row)
// Block: 256 threads    — threads cooperate on the W band columns
// =============================================================================
__global__ void qk_band_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               float* __restrict__ S_band,
                               int N, int D, int W, float scale) {
    const int bh = blockIdx.y;
    const int i  = blockIdx.x;       // query row
    const int half = W / 2;

    // Pointers to this batch_head's Q row and K matrix.
    const float* Q_row = Q + ((size_t)bh * N + i) * D;
    const float* K_bh  = K + (size_t)bh * N * D;
    float* S_row = S_band + ((size_t)bh * N + i) * W;

    // Each thread handles a strided subset of the W band columns.
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        int k = i - half + c;
        if (k < 0 || k >= N) {
            S_row[c] = -FLT_MAX;
        } else {
            float dot = 0.f;
            const float* K_row = K_bh + (size_t)k * D;
            #pragma unroll
            for (int d = 0; d < D; ++d) dot += Q_row[d] * K_row[d];
            S_row[c] = dot * scale;
        }
    }
}

void launch_qk_band(const float* Q, const float* K, float* S_band,
                    int BH, int N, int D, int W, float scale,
                    cudaStream_t stream) {
    dim3 grid(N, BH);
    dim3 block(256);
    qk_band_kernel<<<grid, block, 0, stream>>>(Q, K, S_band, N, D, W, scale);
}

// =============================================================================
// Softmax over each row of the band. Row length = W.
//
// Grid:  (BH * N)       — one block per row of the band
// Block: 256 threads    — cooperate on max-reduction and sum-reduction
// =============================================================================
template <int BLOCK>
__global__ void softmax_band_kernel(float* __restrict__ S_band, int W) {
    const int row = blockIdx.x;
    float* row_ptr = S_band + (size_t)row * W;

    __shared__ float sdata[BLOCK];

    // Pass 1: row max.
    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < W; c += BLOCK) {
        float v = row_ptr[c];
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + off]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // Pass 2: exp, accumulate sum.
    float local_sum = 0.f;
    for (int c = threadIdx.x; c < W; c += BLOCK) {
        // If the entire row is -FLT_MAX (shouldn't happen with non-empty windows),
        // we'd write 0 here and final normalize would 0/0. Guard at the end.
        float e = (row_ptr[c] == -FLT_MAX) ? 0.f : __expf(row_ptr[c] - row_max);
        row_ptr[c] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }
    float row_sum = sdata[0];
    float inv_sum = (row_sum > 0.f) ? (1.f / row_sum) : 0.f;

    // Pass 3: normalize.
    for (int c = threadIdx.x; c < W; c += BLOCK) {
        row_ptr[c] *= inv_sum;
    }
}

void launch_softmax_band(float* S_band, int BH, int N, int W,
                         cudaStream_t stream) {
    constexpr int BLOCK = 256;
    softmax_band_kernel<BLOCK><<<BH * N, BLOCK, 0, stream>>>(S_band, W);
}

// =============================================================================
// Band·V: contract the band against V to produce O.
//
// Grid:  (N, BH)        — one block per (batch_head, query row)
// Block: D threads      — one per output dimension
// =============================================================================
__global__ void band_pv_kernel(const float* __restrict__ S_band,
                               const float* __restrict__ V,
                               float* __restrict__ O,
                               int N, int D, int W) {
    const int bh = blockIdx.y;
    const int i  = blockIdx.x;
    const int d  = threadIdx.x;       // output dimension; assumes blockDim.x == D
    const int half = W / 2;

    if (d >= D) return;

    const float* S_row = S_band + ((size_t)bh * N + i) * W;
    const float* V_bh  = V + (size_t)bh * N * D;
    float* O_row = O + ((size_t)bh * N + i) * D;

    float acc = 0.f;
    for (int c = 0; c < W; ++c) {
        int k = i - half + c;
        if (k < 0 || k >= N) continue;
        // S_row[c] is the softmax probability; if k was OOB, it's already 0
        // (since softmax of -FLT_MAX = 0), but we skip explicitly to avoid
        // reading V out of bounds.
        acc += S_row[c] * V_bh[(size_t)k * D + d];
    }
    O_row[d] = acc;
}

void launch_band_pv(const float* S_band, const float* V, float* O,
                    int BH, int N, int D, int W,
                    cudaStream_t stream) {
    dim3 grid(N, BH);
    dim3 block(D);   // assumes D <= 1024 (true for any realistic head dim)
    band_pv_kernel<<<grid, block, 0, stream>>>(S_band, V, O, N, D, W);
}
