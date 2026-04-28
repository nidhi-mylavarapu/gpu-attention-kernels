#include "kernels.cuh"
#include "common.cuh"

/* 
Fills a buffer with a constant value
Used to initialize ws.S to -FLT_MAX before the tile loop so any entries we skip 
get treated as "masked" by softmax 
*/

__global__ void fill_kernel(float* __restrict__ buf, size_t n, float val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        buf[idx] = val;
    }
}

void launch_fill(float* buf, size_t n, float val, cudaStream_t stream) {
    int block = 256;
    int grid = (int)std::min((size_t)65535, (n + block - 1) / block);
    fill_kernel<<<grid, block, 0, stream>>>(buf, n, val);
}