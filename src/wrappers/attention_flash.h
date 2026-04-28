#pragma once
#include "attention.h"
#include <cublas_v2.h>

// FlashAttention-style fused attention: tiled, with online softmax.
//
// Same signature as attention_forward_naive so the benchmark harness can
// swap implementations transparently.
//
// Workspace requirements differ from the naive version:
//   - Q_bshd, K_bshd, V_bshd: still needed for projection outputs
//   - Q, K, V: still needed (post-transpose, [B, H, S, D] layout)
//   - O: still needed (pre-transpose output)
//   - S: NOT NEEDED — this is the whole point. Allocate as nullptr or skip.
//
// For simplicity we share the AttentionWorkspace struct; ws.S can be left
// unallocated when calling this function, but for a clean A/B benchmark we
// allocate it anyway via allocate_workspace() so peak memory is reported
// honestly per-implementation.
void attention_forward_flash(cublasHandle_t handle,
                             const float* X, const float* Wq,
                             const float* Wk, const float* Wv,
                             float* out_BSD,
                             AttentionWorkspace& ws,
                             const AttentionConfig& cfg,
                             cudaStream_t stream = 0);

// A workspace allocator that omits the [B, H, S, S] tensor. Use this when
// benchmarking the flash kernel to see its true memory footprint.
void allocate_workspace_flash(AttentionWorkspace& ws, const AttentionConfig& cfg);
