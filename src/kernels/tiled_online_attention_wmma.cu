// tiled_online_attention_wmma.cu — Tensor-Core (WMMA) variant of the
// tile-streamed online-softmax attention kernel.
//
// Same algorithm class as src/kernels/tiled_online_attention.cu (FlashAttention-1
// online softmax, no global S matrix). The differences are kernel-engineering:
//   - Q/K/V are cast FP32 -> FP16 once on load into shared memory.
//   - Q*K^T and P*V both run on Tensor Cores via nvcuda::wmma 16x16x16 with
//     FP32 accumulators (mma_sync m16n16k16 fp16 -> fp32).
//   - Online softmax keeps full FP32 precision in shared memory.
//
// Tile layout (specialized for D = 64):
//   Br = 64 (rows of Q processed per block)
//   Bc = 32 (cols of K/V streamed per inner step)
//   D  = 64
//   WARPS_PER_BLOCK = Br / 16 = 4   (each warp owns a 16-row strip of Q)
//   block = (128) threads
//   grid  = (ceil(N / Br), B*H)

#include "common.cuh"
#include "tiled_online_attention_wmma.cuh"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

namespace {

template <int Br, int Bc, int D>
__global__ void tiled_online_attention_wmma_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int N, float scale) {

  using namespace nvcuda;

  static_assert(Br % 16 == 0, "Br must be a multiple of 16");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16");
  static_assert(D  % 16 == 0, "D must be a multiple of 16");
  static_assert(Bc <= 32,
                "softmax mapping assumes one warp lane per S column (Bc <= 32)");

  constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
  constexpr int WARPS  = Br / WMMA_M;
  constexpr int BLOCK  = WARPS * 32;

  constexpr int N_TILES_S = Bc / WMMA_N;   // S = Q*K^T : N-tiles per warp
  constexpr int K_TILES_S = D  / WMMA_K;   // S inner dim
  constexpr int N_TILES_O = D  / WMMA_N;   // O += P*V : N-tiles per warp
  constexpr int K_TILES_O = Bc / WMMA_K;   // O inner dim

  const int lane    = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tid     = threadIdx.x;

  const int bh      = blockIdx.y;
  const int q_tile  = blockIdx.x;

  const int q_row_base    = q_tile * Br;
  const int warp_row_base = q_row_base + warp_id * WMMA_M;

  const float* Qp = Q + (size_t)bh * N * D;
  const float* Kp = K + (size_t)bh * N * D;
  const float* Vp = V + (size_t)bh * N * D;
  float*       Op = O + (size_t)bh * N * D;

  __shared__ __half sQ[Br * D];
  __shared__ __half sK[Bc * D];
  __shared__ __half sV[Bc * D];
  __shared__ float  sS[Br * Bc];
  __shared__ __half sP[Br * Bc];
  __shared__ float  sO[Br * D];
  __shared__ float  sM[Br];
  __shared__ float  sL[Br];

#pragma unroll
  for (int i = tid; i < Br * D; i += BLOCK) sO[i] = 0.f;
  if (tid < Br) {
    sM[tid] = -FLT_MAX;
    sL[tid] = 0.f;
  }

  for (int i = tid; i < Br * D; i += BLOCK) {
    int row = i / D;
    int col = i - row * D;
    int g   = q_row_base + row;
    sQ[i]   = (g < N) ? __float2half(Qp[(size_t)g * D + col])
                      : __float2half(0.f);
  }
  __syncthreads();

  const int num_kv_tiles = (N + Bc - 1) / Bc;
  for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
    const int kv_base = kv_tile * Bc;

    for (int i = tid; i < Bc * D; i += BLOCK) {
      int row = i / D;
      int col = i - row * D;
      int g   = kv_base + row;
      float kf = (g < N) ? Kp[(size_t)g * D + col] : 0.f;
      float vf = (g < N) ? Vp[(size_t)g * D + col] : 0.f;
      sK[i] = __float2half(kf);
      sV[i] = __float2half(vf);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // S = Q * K^T  via WMMA.
    //
    // Q is laid out [Br, D] row-major; treat it as matrix_a row-major (ld=D).
    // K is laid out [Bc, D] row-major; we want S = Q * K^T, which means we
    // need a [D, Bc] matrix as the "B" operand. K^T in [D, Bc] col-major has
    // exactly the same bytes as K in [Bc, D] row-major (ld=D). So we load
    // matrix_b as col_major from sK with ld = D.
    //
    // Each warp computes a 16-row strip of S = [16, Bc] = N_TILES_S x (16x16).
    // ------------------------------------------------------------------
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[N_TILES_S];

#pragma unroll
    for (int n = 0; n < N_TILES_S; ++n) wmma::fill_fragment(s_frag[n], 0.f);

#pragma unroll
    for (int kk = 0; kk < K_TILES_S; ++kk) {
      const __half* a_ptr =
          sQ + (size_t)(warp_id * WMMA_M) * D + kk * WMMA_K;
      wmma::load_matrix_sync(a_frag, a_ptr, D);

#pragma unroll
      for (int nn = 0; nn < N_TILES_S; ++nn) {
        const __half* b_ptr =
            sK + (size_t)(nn * WMMA_N) * D + kk * WMMA_K;
        wmma::load_matrix_sync(b_frag, b_ptr, D);
        wmma::mma_sync(s_frag[nn], a_frag, b_frag, s_frag[nn]);
      }
    }

#pragma unroll
    for (int nn = 0; nn < N_TILES_S; ++nn) {
#pragma unroll
      for (int i = 0; i < (int)s_frag[nn].num_elements; ++i) {
        s_frag[nn].x[i] *= scale;
      }
      float* dst =
          sS + (size_t)(warp_id * WMMA_M) * Bc + nn * WMMA_N;
      wmma::store_matrix_sync(dst, s_frag[nn], Bc, wmma::mem_row_major);
    }

    // No __syncthreads() needed here: each warp owns rows
    //   [warp_id*16, warp_id*16+16) of sS and only reads its own rows in the
    //   softmax stage that follows. The wmma store is warp-synchronous.

    // ------------------------------------------------------------------
    // Online softmax over each warp's 16 rows.
    //
    // 32 warp lanes <-> 32 columns of S (Bc <= 32). For Bc < 32, lanes with
    // lane >= Bc carry s_val = -FLT_MAX so they don't affect max/sum.
    //
    // Per-row state (sM[r], sL[r]) lives in shared. Only lane 0 of the warp
    // updates them; sO[r][:] is rescaled by alpha by all 32 lanes (lane
    // strides over D).
    // ------------------------------------------------------------------
#pragma unroll
    for (int r = 0; r < WMMA_M; ++r) {
      const int row_idx = warp_id * WMMA_M + r;
      const int row_g   = warp_row_base + r;

      float s_val = -FLT_MAX;
      if (lane < Bc) {
        const int g = kv_base + lane;
        if (g < N && row_g < N) {
          s_val = sS[row_idx * Bc + lane];
        }
      }

      float row_max = s_val;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));
      }

      const float m_old = sM[row_idx];
      const float m_new = fmaxf(m_old, row_max);
      const float alpha = (m_old == -FLT_MAX) ? 0.f : __expf(m_old - m_new);

      float p = 0.f;
      if (s_val != -FLT_MAX) {
        p = __expf(s_val - m_new);
      }

      float row_sum = p;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);
      }

      if (lane < Bc) {
        sP[row_idx * Bc + lane] = __float2half(p);
      }

      if (lane == 0) {
        sM[row_idx] = m_new;
        sL[row_idx] = alpha * sL[row_idx] + row_sum;
      }

      if (row_g < N) {
#pragma unroll
        for (int dd = lane; dd < D; dd += 32) {
          sO[row_idx * D + dd] *= alpha;
        }
      }
    }

    __syncthreads();

    // ------------------------------------------------------------------
    // O += P * V  via WMMA.
    //
    // P  = sP : [Br, Bc] FP16 row-major  -> matrix_a row_major (ld = Bc).
    // V  = sV : [Bc, D]  FP16 row-major  -> matrix_b row_major (ld = D).
    // O accumulator is loaded from sO (already alpha-scaled), accumulated
    // into via mma_sync, and stored back.
    // ------------------------------------------------------------------
    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[N_TILES_O];

#pragma unroll
    for (int nn = 0; nn < N_TILES_O; ++nn) {
      const float* o_ptr =
          sO + (size_t)(warp_id * WMMA_M) * D + nn * WMMA_N;
      wmma::load_matrix_sync(o_frag[nn], o_ptr, D, wmma::mem_row_major);
    }

#pragma unroll
    for (int kk = 0; kk < K_TILES_O; ++kk) {
      const __half* p_ptr =
          sP + (size_t)(warp_id * WMMA_M) * Bc + kk * WMMA_K;
      wmma::load_matrix_sync(p_frag, p_ptr, Bc);

#pragma unroll
      for (int nn = 0; nn < N_TILES_O; ++nn) {
        const __half* v_ptr =
            sV + (size_t)(kk * WMMA_K) * D + nn * WMMA_N;
        wmma::load_matrix_sync(v_frag, v_ptr, D);
        wmma::mma_sync(o_frag[nn], p_frag, v_frag, o_frag[nn]);
      }
    }

#pragma unroll
    for (int nn = 0; nn < N_TILES_O; ++nn) {
      float* o_ptr =
          sO + (size_t)(warp_id * WMMA_M) * D + nn * WMMA_N;
      wmma::store_matrix_sync(o_ptr, o_frag[nn], D, wmma::mem_row_major);
    }

    __syncthreads();
  }

  for (int i = tid; i < Br * D; i += BLOCK) {
    int row   = i / D;
    int col   = i - row * D;
    int row_g = q_row_base + row;
    if (row_g < N) {
      const float l     = sL[row];
      const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
      Op[(size_t)row_g * D + col] = sO[i] * inv_l;
    }
  }
}

} // namespace

void launch_tiled_online_attention_wmma_bhsd(const float* Q, const float* K,
                                             const float* V, float* O, int B,
                                             int N, int H, int D, float scale,
                                             cudaStream_t stream) {
  if (D != 64) {
    fprintf(stderr,
            "launch_tiled_online_attention_wmma_bhsd: only D=64 is compiled "
            "(got %d).\n",
            D);
    std::exit(1);
  }

  constexpr int Br = 64;
  constexpr int Bc = 32;
  constexpr int Dt = 64;
  constexpr int BLOCK = (Br / 16) * 32;   // 4 warps

  const int num_q_tiles = (N + Br - 1) / Br;
  dim3 grid(num_q_tiles, B * H);
  dim3 block(BLOCK);

  tiled_online_attention_wmma_kernel<Br, Bc, Dt>
      <<<grid, block, 0, stream>>>(Q, K, V, O, N, scale);
}
