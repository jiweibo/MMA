project(mma LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.18)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/common)
include_directories(/usr/local/include)
include_directories(/usr/local/cuda/include)

add_subdirectory(gemm)
add_subdirectory(reduce)
