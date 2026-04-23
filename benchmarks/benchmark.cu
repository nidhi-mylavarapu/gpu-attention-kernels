#include "attention.h"
#include "common.cuh"
#include "../src/utils/npy_loader.h"

#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>

void cpu_attention_reference(const float*, const float*, const float*,
                             const float*, float*, int, int, int, int);

// ==========================
// Function pointer type
// ==========================
typedef void (*AttentionFn)(
    cublasHandle_t,
    const float*, const float*, const float*, const float*,
    float*,
    AttentionWorkspace&,
    const AttentionConfig&,
    cudaStream_t
);

struct BenchResult {
    int seq_len;
    float mean_ms, min_ms, max_ms;
    size_t workspace_bytes;
    size_t peak_bytes;
    double max_abs_err;
};

// ==========================
// Benchmark ONE sequence length
// ==========================
static BenchResult benchmark_one(cublasHandle_t handle,
                                 const AttentionConfig& cfg,
                                 int warmup, int iters,
                                 bool check_correctness,
                                 AttentionFn attention_fn)
{
    const int Dm = cfg.d_model();
    const size_t BSD = (size_t)cfg.batch * cfg.seq_len * Dm;

    // Load data
    std::string base = "../data/";

    std::vector<float> hX  = load_npy(base + "X_"  + std::to_string(cfg.seq_len) + ".npy");
    std::vector<float> hWq = load_npy(base + "Wq_" + std::to_string(cfg.seq_len) + ".npy");
    std::vector<float> hWk = load_npy(base + "Wk_" + std::to_string(cfg.seq_len) + ".npy");
    std::vector<float> hWv = load_npy(base + "Wv_" + std::to_string(cfg.seq_len) + ".npy");

    // Device memory
    float *dX, *dWq, *dWk, *dWv, *dOut;
    CUDA_CHECK(cudaMalloc(&dX,  BSD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWq, hWq.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWk, hWk.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWv, hWv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, BSD * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  BSD * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), hWq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWk, hWk.data(), hWk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWv, hWv.data(), hWv.size() * sizeof(float), cudaMemcpyHostToDevice));

    AttentionWorkspace ws{};
    allocate_workspace(ws, cfg);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        attention_fn(handle, dX, dWq, dWk, dWv, dOut, ws, cfg, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times_ms(iters);
    GpuTimer t;

    for (int i = 0; i < iters; ++i) {
        t.start();
        attention_fn(handle, dX, dWq, dWk, dWv, dOut, ws, cfg, 0);
        times_ms[i] = t.stop();
    }

    size_t mem_peak = gpu_mem_used_bytes();

    BenchResult r{};
    r.seq_len = cfg.seq_len;
    r.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

    float sum = 0.f;
    for (float x : times_ms) sum += x;
    r.mean_ms = sum / iters;

    r.workspace_bytes = ws.total_bytes;
    r.peak_bytes = mem_peak;
    r.max_abs_err = std::nan("");

    // Correctness
    if (check_correctness) {
        std::vector<float> hOut(BSD), hRef(BSD);

        CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, BSD * sizeof(float),
                              cudaMemcpyDeviceToHost));

        cpu_attention_reference(hX.data(), hWq.data(), hWk.data(), hWv.data(),
                                hRef.data(),
                                cfg.batch, cfg.seq_len, cfg.n_heads, cfg.d_head);

        double max_err = 0.0;
        for (size_t i = 0; i < BSD; ++i) {
            max_err = std::max(max_err,
                               (double)std::fabs(hOut[i] - hRef[i]));
        }
        r.max_abs_err = max_err;
    }

    free_workspace(ws);
    cudaFree(dX);
    cudaFree(dWq);
    cudaFree(dWk);
    cudaFree(dWv);
    cudaFree(dOut);

    return r;
}

// ==========================
// MAIN
// ==========================
int main(int argc, char** argv) {
    std::string kernel = "naive";
    if (argc > 1) kernel = argv[1];

    AttentionFn attention_fn = nullptr;

    if (kernel == "naive") {
        attention_fn = attention_forward_naive;
    } else if (kernel == "tiled") {
        attention_fn = attention_forward_tiled;
    } else if (kernel == "fused") {
        attention_fn = attention_forward_fused;
    } else {
        printf("Unknown kernel: %s\n", kernel.c_str());
        return 1;
    }

    AttentionConfig base{};
    base.batch   = 1;
    base.n_heads = 8;
    base.d_head  = 64;

    const std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096};
    const int warmup = 3;
    const int iters  = 10;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("kernel=%s\n", kernel.c_str());
    printf("seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err\n");

    for (int S : seq_lens) {
        AttentionConfig cfg = base;
        cfg.seq_len = S;

        bool check = (S <= 512);

        BenchResult r = benchmark_one(handle, cfg, warmup, iters, check, attention_fn);

        printf("%d,%.3f,%.3f,%.3f,%.1f,%.1f,%s\n",
               r.seq_len,
               r.mean_ms,
               r.min_ms,
               r.max_ms,
               r.workspace_bytes / (1024.0 * 1024.0),
               r.peak_bytes     / (1024.0 * 1024.0),
               std::isnan(r.max_abs_err) ? "skip" :
                   std::to_string(r.max_abs_err).c_str());
    }

    cublasDestroy(handle);
    return 0;
}