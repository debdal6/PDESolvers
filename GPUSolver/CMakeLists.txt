cmake_minimum_required(VERSION 3.22)
project(GPUSolver)

set(CMAKE_CXX_STANDARD 17)
set(CPPVERSION 17)
set(SM_ARCH 86)

set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})
set(CMAKE_CUDA_STANDARD ${CPPVERSION})
set(CMAKE_CXX_STANDARD ${CPPVERSION})
set(CMAKE_CXX_FLAGS "-std=c++${CPPVERSION}")
set(CMAKE_CUDA_FLAGS "-std=c++${CPPVERSION}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-std=c++${CPPVERSION}")
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
add_executable(GPU_runner_explicit)
add_executable(CPU_runner_explicit)

target_link_libraries(GPU_runner_explicit PRIVATE
        CUDA::cusparse
        CUDA::cudart
)

target_sources(CPU_runner_explicit
        PRIVATE
        cpu/bse_solvers.cpp
        cpu/bse_solvers.h
)
target_sources(GPU_runner_explicit
        PRIVATE
#        gpu/bse_explicit_parallel.cu
#        playground/cn_playground.cu
#        playground/sample.cu
#        playground/bse_cn.cu
        gpu/bse_solvers_parallel.cuh
        main.cu
)

set_target_properties(GPU_runner_explicit
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(CPU_runner_explicit
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
