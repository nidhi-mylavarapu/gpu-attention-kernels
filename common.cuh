#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err)); std::exit(1);                  \
    }                                                                    \
} while (0)

#define CUBLAS_CHECK(call) do {                                          \
    cublasStatus_t s = (call);                                           \
    if (s != CUBLAS_STATUS_SUCCESS) {                                    \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,  \
                (int)s); std::exit(1);                                   \
    }                                                                    \
} while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_); }
    float stop() {                             // returns milliseconds
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

inline size_t gpu_mem_used_bytes() {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return total_b - free_b;
}