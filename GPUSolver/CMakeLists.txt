cmake_minimum_required(VERSION 3.22)
project(GPUSolver)

set(CMAKE_CXX_STANDARD 20)
set(CPPVERSION 20)
set(SM_ARCH 86)

set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})
set(CMAKE_CUDA_STANDARD ${CPPVERSION})
set(CMAKE_CXX_STANDARD ${CPPVERSION})
set(CMAKE_CXX_FLAGS "-std=c++${CPPVERSION}")
set(CMAKE_CUDA_FLAGS "-std=c++${CPPVERSION}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-std=c++${CPPVERSION}")

enable_language(CUDA)
add_executable(mymain)

target_sources(mymain
        PRIVATE
#        gpu/gbm_parallel.cu
#        cpu/gbm.cpp
#        cpu/bse_solvers.cpp
#        cpu/bse_solvers.h
#        main.cpp
        gpu/bse_explicit_parallel.cu
#        gpu/test.cu
)

set_target_properties(mymain
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
