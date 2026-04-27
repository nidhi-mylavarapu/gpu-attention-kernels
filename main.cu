#include "attention.h"
#include "attention_flash.h"
#include "common.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>

void cpu_attention_reference(const float*, const float*, const float*,
                             const float*, float*, int, int, int, int);

static void fill_randn(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 0.02f);
    for (auto& x : v) x = dist(rng);
}

enum class Impl { Naive, Flash };

struct BenchResult {
    int seq_len;
    Impl impl;
    float mean_ms, min_ms, max_ms;
    size_t workspace_bytes;
    size_t peak_bytes;
    double max_abs_err;
};

static const char* impl_name(Impl i) {
    return i == Impl::Naive ? "naive" : "flash";
}

static BenchResult benchmark_one(cublasHandle_t handle,
                                 const AttentionConfig& cfg,
                                 Impl impl,
                                 int warmup, int iters,
                                 bool check_correctness)
{
    const int Dm = cfg.d_model();
    const size_t BSD = (size_t)cfg.batch * cfg.seq_len * Dm;
    const size_t WSZ = (size_t)Dm * Dm;

    std::vector<float> hX(BSD), hWq(WSZ), hWk(WSZ), hWv(WSZ);
    fill_randn(hX, 1); fill_randn(hWq, 2); fill_randn(hWk, 3); fill_randn(hWv, 4);

    float *dX, *dWq, *dWk, *dWv, *dOut;
    CUDA_CHECK(cudaMalloc(&dX,  BSD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWq, WSZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWk, WSZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWv, WSZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, BSD * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  BSD * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), WSZ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWk, hWk.data(), WSZ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWv, hWv.data(), WSZ * sizeof(float), cudaMemcpyHostToDevice));

    AttentionWorkspace ws{};
    if (impl == Impl::Naive) allocate_workspace(ws, cfg);
    else                     allocate_workspace_flash(ws, cfg);

    auto run_once = [&]() {
        if (impl == Impl::Naive)
            attention_forward_naive(handle, dX, dWq, dWk, dWv, dOut, ws, cfg);
        else
            attention_forward_flash(handle, dX, dWq, dWk, dWv, dOut, ws, cfg);
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    std::vector<float> times_ms(iters);
    GpuTimer t;
    for (int i = 0; i < iters; ++i) {
        t.start();
        run_once();
        times_ms[i] = t.stop();
    }
    size_t mem_peak = gpu_mem_used_bytes();

    BenchResult r{};
    r.seq_len = cfg.seq_len;
    r.impl = impl;
    r.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = *std::max_element(times_ms.begin(), times_ms.end());
    float sum = 0.f; for (float x : times_ms) sum += x;
    r.mean_ms = sum / iters;
    r.workspace_bytes = ws.total_bytes;
    r.peak_bytes = mem_peak;
    r.max_abs_err = std::nan("");

    if (check_correctness) {
        std::vector<float> hOut(BSD), hRef(BSD);
        CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, BSD * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cpu_attention_reference(hX.data(), hWq.data(), hWk.data(), hWv.data(),
                                hRef.data(),
                                cfg.batch, cfg.seq_len, cfg.n_heads, cfg.d_head);
        double max_err = 0.0;
        for (size_t i = 0; i < BSD; ++i)
            max_err = std::max(max_err, (double)std::fabs(hOut[i] - hRef[i]));
        r.max_abs_err = max_err;
    }

    free_workspace(ws);
    cudaFree(dX); cudaFree(dWq); cudaFree(dWk); cudaFree(dWv); cudaFree(dOut);
    return r;
}

int main(int argc, char** argv) {
    AttentionConfig base{};
    base.batch   = 1;
    base.n_heads = 8;
    base.d_head  = 64;

    const std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup = 3, iters = 10;

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

    printf("impl,seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err\n");
    for (Impl impl : {Impl::Naive, Impl::Flash}) {
        for (int S : seq_lens) {
            AttentionConfig cfg = base; cfg.seq_len = S;
            bool check = (S <= 512);
            BenchResult r = benchmark_one(handle, cfg, impl, warmup, iters, check);
            printf("%s,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%s\n",
                   impl_name(r.impl), r.seq_len, r.mean_ms, r.min_ms, r.max_ms,
                   r.workspace_bytes / (1024.0 * 1024.0),
                   r.peak_bytes     / (1024.0 * 1024.0),
                   std::isnan(r.max_abs_err) ? "skip" :
                       std::to_string(r.max_abs_err).c_str());
        }
    }
    cublasDestroy(handle);
    return 0;
}
