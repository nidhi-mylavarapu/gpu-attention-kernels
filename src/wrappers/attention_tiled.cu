#include "attention.h"
#include <cstdio>

void attention_forward_tiled(
    cublasHandle_t handle,
    const float* X, const float* Wq,
    const float* Wk, const float* Wv,
    float* out,
    AttentionWorkspace& ws,
    const AttentionConfig& cfg,
    cudaStream_t stream)
{
    printf("Tiled not implemented → using naive\n");
    attention_forward_naive(handle, X, Wq, Wk, Wv, out, ws, cfg, stream);
}