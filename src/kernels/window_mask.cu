#include "kernels.cuh"
#include "common.cuh"
#include <cfloat> // for FLT_MAX

// Window mask kernel
/* 
Input: stack of score matrices S of shape [n_mats, rows, row_len]
In our attention pipeline this is the QK^T result before softmax with
shape [batch * n_heads, seq_len, seq_len]
Each (batch,head) pair is one "matrix"; rows are query positions i; columns 
key positions j; entry S[mat, i j] is the unnormalized attention score for query i 
and key j for that batch and head.

Goal: zero out (after the sofmax) any (i,j) where |i-j| > half_window, so each 
query only attends to keys within a sliding window around it. 
We do this by writing -FLT_MAX into masked entries; softmax then does
exp(-FLT_MAX-row_max) = 0, so they don't contribute to the output

Parallelization: one thread per (mat, i, j) entry of the score matrix
the grid is 3d - blockIdx.x for j (with threads inside the block handling
consecutive columns), blockIdx.y for i, and blockIdx.z for mat.

 */

__global__ void window_mask_kernel(float* __restrict__ S,
                                   int rows_per_mat, // seq_len (number of queries i)
                                   int row_len, // seq_len (number of keys j)
                                   int half_window) { // window_size / 2
    // decode this thread's (mat, i, j) coordinates from the grid
    int row = blockIdx.y; // query index i 
    int mat = blockIdx.z; // batch * head index 
    int col = blockIdx.x * blockDim.x + threadIdx.x; // key index j
    if (col >= row_len) return;

    // the grid in x is rounded up to cove row_len so the last block has
    // some threads with col >= row_len; they do nothing
    int diff = col - row;
    if (diff < 0) diff = -diff;
    
    // compute |i-j| 
    if (diff > half_window) {
        size_t idx = ((size_t)mat * rows_per_mat + row) * row_len + col;
        S[idx] = -FLT_MAX;
    }
}

void launch_window_mask(float* S, int n_mats, int rows, int row_len,
                        int window_size, cudaStream_t stream) {
    int half = window_size / 2;
    int block = 256;
    // 3D grid:
    //   x: ceil(row_len / block) blocks across the column axis
    //   y: one block per row (query position i)
    //   z: one block per matrix (batch * head)
    // Total threads ≈ n_mats * rows * row_len, one per score entry.
    dim3 grid((row_len + block - 1) / block, rows, n_mats);
    window_mask_kernel<<<grid, block, 0, stream>>>(S, rows, row_len, half);
}