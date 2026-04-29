// attention.h
#pragma once
#include <cublas_v2.h>

struct AttentionConfig {
    int batch;
    int seq_len;
    int n_heads;
    int d_head;
    int window_size = 0; // 0 = no masking / full attention 
    int d_model() const { return n_heads * d_head; }
};

// Workspace holds all intermediate device buffers so they can be reused
// across timed iterations without allocation overhead.
struct AttentionWorkspace {
    float* Q_bshd = nullptr;    // [B, S, H, D]  projection output
    float* K_bshd = nullptr;
    float* V_bshd = nullptr;
    float* Q = nullptr;         // [B, H, S, D]  after transpose
    float* K = nullptr;
    float* V = nullptr;
    float* S = nullptr;         // dense: [B,H,S,S]; sparse_window: [B,H,S,W] band
    float* O = nullptr;         // [B, H, S, D]  attention output (pre-reshape)
    size_t total_bytes = 0;
};

void allocate_workspace(AttentionWorkspace& ws, const AttentionConfig& cfg);
void allocate_workspace_tiled_online(AttentionWorkspace& ws, const AttentionConfig& cfg);
void free_workspace(AttentionWorkspace& ws);

// X:    [B, S, D_model]
// W_*:  [D_model, D_model]
// Out:  [B, S, D_model]  (written to out_BSD)
void attention_forward_naive(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);

void attention_forward_tiled(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);

void attention_forward_tiled_online(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);

// Same workspace + same outer pipeline as attention_forward_tiled_online,
// but the attention core uses Tensor Cores (WMMA, FP16 inputs, FP32 accum).
void attention_forward_tiled_online_wmma(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);

void attention_forward_sparse_window(
    cublasHandle_t handle,
    const float* X, const float* Wq, const float* Wk, const float* Wv,
    float* Out,
    AttentionWorkspace& ws,
    const AttentionConfig& cfg,
    cudaStream_t stream
);

void allocate_workspace_sparse_window(
    AttentionWorkspace& ws,
    const AttentionConfig& cfg,
    int window_size);