// attention.h
#pragma once
#include <cublas_v2.h>

struct AttentionConfig {
    int batch;
    int seq_len;
    int n_heads;
    int d_head;
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
    float* S = nullptr;         // [B, H, S, S]  attention scores
    float* O = nullptr;         // [B, H, S, D]  attention output (pre-reshape)
    size_t total_bytes = 0;
};

void allocate_workspace(AttentionWorkspace& ws, const AttentionConfig& cfg);
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

void attention_forward_fused(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);