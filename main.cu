#include "src/wrappers/attention.h"
#include "src/wrappers/attention_banded_window.h"
#include "src/kernels/common.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <fstream>
#include <sys/stat.h>

void cpu_attention_reference(const float*, const float*, const float*,
                             const float*, float*, int, int, int, int, int);

static void fill_randn(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 0.02f);
    for (auto& x : v) x = dist(rng);
}

// Write a flat float buffer to disk. Format is just raw float32, no header —
// shape is implied by the filename and known to both sides of the benchmark.
static void dump_floats(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path.c_str()); std::exit(1); }
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

static bool file_exists(const std::string& path) {
    struct stat s;
    return stat(path.c_str(), &s) == 0;
}

enum class Impl { Naive, TiledOnline, BandedWindow };

struct BenchResult {
    int seq_len;
    Impl impl;
    float mean_ms, min_ms, max_ms;
    size_t workspace_bytes;
    size_t peak_bytes;
    double max_abs_err;
};

static const char* impl_name(Impl i) {
    switch (i) {
        case Impl::Naive:        return "naive_cublas";
        case Impl::TiledOnline:   return "tiled_online";
        case Impl::BandedWindow: return "banded_window";
    }
    return "unknown";
}

// Generate or load tensors for this seq_len. Always returns the same bytes
// across runs (deterministic from seed), and writes them to disk so the
// Python benchmark can load identical data.
static void prepare_tensors(const AttentionConfig& cfg,
                            const std::string& tensor_dir,
                            std::vector<float>& hX,
                            std::vector<float>& hWq,
                            std::vector<float>& hWk,
                            std::vector<float>& hWv) {
    const int Dm = cfg.d_model();
    const size_t BSD = (size_t)cfg.batch * cfg.seq_len * Dm;
    const size_t WSZ = (size_t)Dm * Dm;

    hX.resize(BSD); hWq.resize(WSZ); hWk.resize(WSZ); hWv.resize(WSZ);

    // Seed depends on seq_len so each shape gets distinct (but reproducible) data.
    fill_randn(hX,  100u + cfg.seq_len);
    fill_randn(hWq, 200u + cfg.seq_len);
    fill_randn(hWk, 300u + cfg.seq_len);
    fill_randn(hWv, 400u + cfg.seq_len);

    // Filename includes shape so it's unambiguous which file matches which config.
    char base[256];
    snprintf(base, sizeof(base), "%s/n%d_b%d_h%d_d%d",
             tensor_dir.c_str(), cfg.seq_len, cfg.batch, cfg.n_heads, cfg.d_head);

    std::string fX  = std::string(base) + "_X.bin";
    std::string fWq = std::string(base) + "_Wq.bin";
    std::string fWk = std::string(base) + "_Wk.bin";
    std::string fWv = std::string(base) + "_Wv.bin";

    if (!file_exists(fX))  dump_floats(fX,  hX);
    if (!file_exists(fWq)) dump_floats(fWq, hWq);
    if (!file_exists(fWk)) dump_floats(fWk, hWk);
    if (!file_exists(fWv)) dump_floats(fWv, hWv);
}

static BenchResult benchmark_one(cublasHandle_t handle,
                                 const AttentionConfig& cfg,
                                 Impl impl,
                                 int warmup, int iters,
                                 bool check_correctness,
                                 const std::string& tensor_dir)
{
    const int Dm = cfg.d_model();
    const size_t BSD = (size_t)cfg.batch * cfg.seq_len * Dm;
    const size_t WSZ = (size_t)Dm * Dm;

    std::vector<float> hX, hWq, hWk, hWv;
    prepare_tensors(cfg, tensor_dir, hX, hWq, hWk, hWv);

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
    if (impl == Impl::Naive) {
        allocate_workspace(ws, cfg);
    } else if (impl == Impl::TiledOnline) {
        allocate_workspace_tiled_online(ws, cfg);
    } else {
        allocate_workspace_banded(ws, cfg, cfg.window_size);
    }

    auto run_once = [&]() {
        if (impl == Impl::Naive) {
            attention_forward_naive(handle, dX, dWq, dWk, dWv, dOut, ws, cfg);
        } else if (impl == Impl::TiledOnline) {
            attention_forward_tiled_online(handle, dX, dWq, dWk, dWv, dOut, ws, cfg);
        } else if (impl == Impl::BandedWindow) {
            attention_forward_banded_window(handle, dX, dWq, dWk, dWv, dOut, ws, cfg);
        }
    };
    for (int i = 0; i < warmup; ++i) run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

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
                                cfg.batch, cfg.seq_len, cfg.n_heads, cfg.d_head, 0);
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
    base.window_size = 256;   // sliding window width


    const std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup = 3, iters = 10;

    std::string tensor_dir = "tensors";
    if (argc > 1) tensor_dir = argv[1];
    mkdir(tensor_dir.c_str(), 0755);  // ignore EEXIST

    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

    printf("impl,seq_len,mean_ms,min_ms,max_ms,workspace_MB,peak_MB,max_abs_err\n");
    for (Impl impl : {Impl::Naive, Impl::TiledOnline, Impl::BandedWindow}) {
        for (int S : seq_lens) {
            AttentionConfig cfg = base; cfg.seq_len = S;
            bool check = (S <= 512) && (impl != Impl::BandedWindow);
            BenchResult r = benchmark_one(handle, cfg, impl, warmup, iters, check, tensor_dir);
            printf("%s,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%s\n",
                   impl_name(r.impl), r.seq_len, r.mean_ms, r.min_ms, r.max_ms,
                   r.workspace_bytes / (1024.0 * 1024.0),
                   r.peak_bytes     / (1024.0 * 1024.0),
                   std::isnan(r.max_abs_err) ? "skip" :
                       std::to_string(r.max_abs_err).c_str());
        }
    }
    cublasDestroy(handle);
    fprintf(stderr,
    "\nTensors written to %s/. Run bench_flash_attn.py to benchmark "
    "official flash-attn on the same data.\n", tensor_dir.c_str());
    return 0;
}

