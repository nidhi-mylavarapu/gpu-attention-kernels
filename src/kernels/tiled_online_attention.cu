// tiled_online_attention.cu — tile-streamed attention with online softmax.
//
// Same algorithm class as FlashAttention-1 (tile-streamed Q/K/V with online
// softmax, no global S matrix), implemented with a warp-centric thread
// mapping (see Mapping section below).
//
// Mapping
// -------
// Grid:  (ceil(N / Br), B*H)         — one block per (Q tile, batch_head)
// Block: (32, Br) threads            — one WARP per Q row in the tile
//                                      (lane = threadIdx.x, warp = threadIdx.y)
//
// Why warp-per-row? It lets us:
//   - reduce row max / row sum with __shfl_xor_sync (no shared scratch),
//   - distribute the O accumulator across the warp lanes (each lane keeps
//     D/32 partial outputs in registers) instead of one thread holding a
//     full D-wide accumulator,
//   - cooperatively load Q, K, V tiles using all BLOCK threads.
//
// Online softmax
// --------------
// Maintain per-row running statistics (one warp = one row):
//   m_i  : running max of pre-softmax scores seen so far
//   l_i  : running sum of exp(s - m_i)  over scores seen so far
//   O_i  : running unnormalized weighted-V output
//
// On a new K/V tile that produces partial scores S_ij with row max m':
//   m_new = max(m_i, m')
//   alpha = exp(m_i - m_new)               (rescales prior accumulators)
//   P_ij  = exp(S_ij - m_new)
//   l_new = alpha * l_i + sum_j P_ij
//   O_new = alpha * O_i + P_ij * V_j
// Final O is O_last / l_last.

#include "common.cuh"
#include "tiled_online_attention.cuh"
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace {

template <int Br, int Bc, int D>
__global__ void tiled_online_attention_kernel(const float *__restrict__ Q,
                                              const float *__restrict__ K,
                                              const float *__restrict__ V,
                                              float *__restrict__ O, int N,
                                              float scale) {
  static_assert(D % 32 == 0, "D must be a multiple of warp size (32).");

  constexpr int WARP = 32;
  constexpr int WARPS = Br; // one warp per Q row in the tile
  constexpr int BLOCK = WARPS * WARP;
  constexpr int DPL = D / WARP;               // D dims owned per lane
  constexpr int SPL = (Bc + WARP - 1) / WARP; // score slots per lane

  const int lane = threadIdx.x;    // 0..31
  const int warp_id = threadIdx.y; // 0..WARPS-1
  const int tid = warp_id * WARP + lane;

  const int bh = blockIdx.y;
  const int q_tile = blockIdx.x;

  const int q_row_global = q_tile * Br + warp_id;
  const bool row_valid = (q_row_global < N);

  const float *Qp = Q + (size_t)bh * N * D;
  const float *Kp = K + (size_t)bh * N * D;
  const float *Vp = V + (size_t)bh * N * D;
  float *Op = O + (size_t)bh * N * D;

  __shared__ float sQ[Br * D];
  __shared__ float sK[Bc * D];
  __shared__ float sV[Bc * D];
  __shared__ float sP[Br * Bc];

  {
    constexpr int N_ELEM = Br * D;
#pragma unroll
    for (int i = tid; i < N_ELEM; i += BLOCK) {
      int row = i / D;
      int col = i - row * D;
      int g = q_tile * Br + row;
      sQ[i] = (g < N) ? Qp[(size_t)g * D + col] : 0.f;
    }
  }

  float m_i = -FLT_MAX;
  float l_i = 0.f;

  float O_reg[DPL];
#pragma unroll
  for (int j = 0; j < DPL; ++j)
    O_reg[j] = 0.f;

  __syncthreads();

  const int num_kv_tiles = (N + Bc - 1) / Bc;
  for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
    const int kv_base = kv_tile * Bc;

    {
      constexpr int N_ELEM = Bc * D;
#pragma unroll
      for (int i = tid; i < N_ELEM; i += BLOCK) {
        int row = i / D;
        int col = i - row * D;
        int g = kv_base + row;
        if (g < N) {
          sK[i] = Kp[(size_t)g * D + col];
          sV[i] = Vp[(size_t)g * D + col];
        } else {
          sK[i] = 0.f;
          sV[i] = 0.f;
        }
      }
    }
    __syncthreads();

    float S_lane[SPL];
    float row_max_local = -FLT_MAX;

#pragma unroll
    for (int s = 0; s < SPL; ++s) {
      const int c = lane + s * WARP;
      float dot = -FLT_MAX;
      if (c < Bc) {
        const int g = kv_base + c;
        if (g < N && row_valid) {
          float acc = 0.f;
#pragma unroll
          for (int d = 0; d < D; ++d) {
            acc += sQ[warp_id * D + d] * sK[c * D + d];
          }
          dot = acc * scale;
        }
      }
      S_lane[s] = dot;
      if (dot > row_max_local)
        row_max_local = dot;
    }

#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
      float v = __shfl_xor_sync(0xffffffff, row_max_local, off);
      row_max_local = fmaxf(row_max_local, v);
    }

    const float m_new = fmaxf(m_i, row_max_local);
    const float alpha = (m_i == -FLT_MAX) ? 0.f : __expf(m_i - m_new);

    float row_sum_local = 0.f;
#pragma unroll
    for (int s = 0; s < SPL; ++s) {
      const int c = lane + s * WARP;
      float p = 0.f;
      if (c < Bc && S_lane[s] != -FLT_MAX) {
        p = __expf(S_lane[s] - m_new);
      }
      if (c < Bc)
        sP[warp_id * Bc + c] = p;
      row_sum_local += p;
    }

#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
      row_sum_local += __shfl_xor_sync(0xffffffff, row_sum_local, off);
    }

    l_i = alpha * l_i + row_sum_local;

    __syncwarp();

#pragma unroll
    for (int j = 0; j < DPL; ++j) {
      const int d = lane + j * WARP;
      float pv = 0.f;
#pragma unroll
      for (int c = 0; c < Bc; ++c) {
        pv += sP[warp_id * Bc + c] * sV[c * D + d];
      }
      O_reg[j] = alpha * O_reg[j] + pv;
    }

    m_i = m_new;
    __syncthreads();
  }

  if (row_valid) {
    const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
#pragma unroll
    for (int j = 0; j < DPL; ++j) {
      const int d = lane + j * 32;
      Op[(size_t)q_row_global * D + d] = O_reg[j] * inv_l;
    }
  }
}

} // namespace

void launch_tiled_online_attention_bhsd(const float *Q, const float *K,
                                        const float *V, float *O, int B, int N,
                                        int H, int D, float scale,
                                        cudaStream_t stream) {
  if (D != 64) {
    fprintf(
        stderr,
        "launch_tiled_online_attention_bhsd: only D=64 is compiled (got %d). "
        "Add a template instantiation for other head dims.\n",
        D);
    std::exit(1);
  }

  constexpr int Br = 8;
  constexpr int Bc = 64;
  constexpr int Dt = 64;

  const int num_q_tiles = (N + Br - 1) / Br;
  dim3 grid(num_q_tiles, B * H);
  dim3 block(32, Br);

  tiled_online_attention_kernel<Br, Bc, Dt>
      <<<grid, block, 0, stream>>>(Q, K, V, O, N, scale);
}
