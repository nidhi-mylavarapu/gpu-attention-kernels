// reference.cpp - CPU attention for correctness checks (small seq only)
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

void cpu_attention_reference(
    const float* X, const float* Wq, const float* Wk, const float* Wv,
    float* out,
    int B, int S, int H, int D,
    int window_size)        // 0 = no masking (full attention)
{
    int Dm = H * D;
    auto Xm  = [&](int b,int s,int d){ return X[((size_t)b*S+s)*Dm+d]; };
    auto Wqm = [&](int i,int j){ return Wq[(size_t)i*Dm+j]; };
    auto Wkm = [&](int i,int j){ return Wk[(size_t)i*Dm+j]; };
    auto Wvm = [&](int i,int j){ return Wv[(size_t)i*Dm+j]; };

    std::vector<float> Q((size_t)B*S*Dm), K((size_t)B*S*Dm), V((size_t)B*S*Dm);
    // Q = X @ Wq, etc.
    for (int b = 0; b < B; ++b)
      for (int s = 0; s < S; ++s)
        for (int j = 0; j < Dm; ++j) {
            float q=0, k=0, v=0;
            for (int i = 0; i < Dm; ++i) {
                float x = Xm(b,s,i);
                q += x * Wqm(i,j); k += x * Wkm(i,j); v += x * Wvm(i,j);
            }
            Q[((size_t)b*S+s)*Dm+j] = q;
            K[((size_t)b*S+s)*Dm+j] = k;
            V[((size_t)b*S+s)*Dm+j] = v;
        }

    float scale = 1.f / std::sqrt((float)D);
    std::vector<float> row(S);

    const int half = window_size / 2;
    const bool use_window = (window_size > 0);

    for (int b = 0; b < B; ++b)
     for (int h = 0; h < H; ++h)
      for (int i = 0; i < S; ++i) {
        // scores row_i = Q[b,i,h,:] · K[b,j,h,:]  for all j
        float rmax = -INFINITY;
        for (int j = 0; j < S; ++j) {
            // Window mask: same convention as the GPU kernel.
            if (use_window && std::abs(i - j) > half) {
                row[j] = -INFINITY;
                continue;
            }
            float s = 0.f;
            for (int d = 0; d < D; ++d) {
                float qv = Q[((size_t)b*S+i)*Dm + h*D + d];
                float kv = K[((size_t)b*S+j)*Dm + h*D + d];
                s += qv * kv;
            }
            s *= scale;
            row[j] = s;
            if (s > rmax) rmax = s;
        }
        float sum = 0.f;
        for (int j = 0; j < S; ++j) {
            // exp(-inf - rmax) = 0, so masked entries contribute nothing.
            row[j] = std::exp(row[j]-rmax);
            sum += row[j];
        }
        for (int j = 0; j < S; ++j) row[j] /= sum;

        // out[b, i, h, :] = sum_j row[j] * V[b, j, h, :]
        for (int d = 0; d < D; ++d) {
            float o = 0.f;
            for (int j = 0; j < S; ++j)
                o += row[j] * V[((size_t)b*S+j)*Dm + h*D + d];
            out[((size_t)b*S+i)*Dm + h*D + d] = o;
        }
      }
}