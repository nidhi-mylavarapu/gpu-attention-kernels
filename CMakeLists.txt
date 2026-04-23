cmake_minimum_required(VERSION 3.18)
project(attention_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

find_package(CUDAToolkit REQUIRED)

add_executable(bench
    benchmarks/benchmark.cu
    src/wrappers/attention_naive.cu
    src/wrappers/attention_tiled.cu
    src/wrappers/attention_fused.cu
    src/kernels/kernels.cu
    src/reference.cpp
)
target_include_directories(bench PRIVATE
    include
    src/kernels
    src/utils
)
target_link_libraries(bench PRIVATE CUDA::cublas CUDA::cudart)
target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>)

    