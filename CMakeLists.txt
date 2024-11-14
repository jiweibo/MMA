cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "61;75;86")

project(mma LANGUAGES CXX CUDA)

set(CUTLASS_DIR
    "/wilber/repo/cutlass/build/install"
    CACHE STRING "cutlass install dir")
set(CUDA_DIR
    "/usr/local/cuda"
    CACHE STRING "cuda dir")

include(cmake/third_party.cmake)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/local/include)
include_directories(${CUDA_DIR}/include)
include_directories(${CUTLASS_DIR}/include)

link_directories(${CUDA_DIR}/lib64)
link_directories(${CUTLASS_DIR}/lib)

set(DEPS glog gflags gtest unwind)

add_subdirectory(gemm)
add_subdirectory(reduce)
add_subdirectory(ops)

enable_testing()
add_subdirectory(tests)
