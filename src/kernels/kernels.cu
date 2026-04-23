// kernels.cu
#include "kernels.cuh"
#include "common.cuh"
#include <cfloat>

// ---------- Softmax ----------
// One block per row. Threads cooperate on max-reduction and sum-reduction.
// Block size is 256; each thread strides through the row.
template <int BLOCK>
__global__ void softmax_scaled_kernel(float* __restrict__ S,
                                      int row_len, float scale) {
    int row = blockIdx.x;
    float* row_ptr = S + (size_t)row * row_len;

    // --- pass 1: find max (after scaling)
    __shared__ float sdata[BLOCK];
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_len; i += BLOCK) {
        float v = row_ptr[i] * scale;
        row_ptr[i] = v;                 // write back scaled value
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                       sdata[threadIdx.x + off]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // --- pass 2: exp(x - max), accumulate sum
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < row_len; i += BLOCK) {
        float e = __expf(row_ptr[i] - row_max);
        row_ptr[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }
    float inv_sum = 1.f / sdata[0];

    // --- pass 3: normalize
    for (int i = threadIdx.x; i < row_len; i += BLOCK) {
        row_ptr[i] *= inv_sum;
    }
}

void launch_softmax_scaled(float* S, int rows, int row_len, float scale,
                           cudaStream_t stream) {
    constexpr int BLOCK = 256;
    softmax_scaled_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(S, row_len, scale);
}

// ---------- Transpose ----------
__global__ void transpose_bshd_to_bhsd_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int B, int S, int H, int D) {
    // Flat idx across [B, S, H, D]
    size_t total = (size_t)B * S * H * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int d = idx % D;
        int h = (idx / D) % H;
        int s = (idx / ((size_t)D * H)) % S;
        int b = idx / ((size_t)D * H * S);
        size_t dst_idx = (((size_t)b * H + h) * S + s) * D + d;
        dst[dst_idx] = src[idx];
    }
}

__global__ void transpose_bhsd_to_bshd_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int B, int H, int S, int D) {
    size_t total = (size_t)B * H * S * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int d = idx % D;
        int s = (idx / D) % S;
        int h = (idx / ((size_t)D * S)) % H;
        int b = idx / ((size_t)D * S * H);
        size_t dst_idx = (((size_t)b * S + s) * H + h) * D + d;
        dst[dst_idx] = src[idx];
    }
}

void launch_transpose_bshd_to_bhsd(const float* src, float* dst,
                                   int B, int S, int H, int D,
                                   cudaStream_t stream) {
    int block = 256;
    int grid = std::min((int)(((size_t)B*S*H*D + block - 1) / block), 65535);
    transpose_bshd_to_bhsd_kernel<<<grid, block, 0, stream>>>(src, dst, B, S, H, D);
}

void launch_transpose_bhsd_to_bshd(const float* src, float* dst,
                                   int B, int H, int S, int D,
                                   cudaStream_t stream) {
    int block = 256;
    int grid = std::min((int)(((size_t)B*H*S*D + block - 1) / block), 65535);
    transpose_bhsd_to_bshd_kernel<<<grid, block, 0, stream>>>(src, dst, B, H, S, D);
}