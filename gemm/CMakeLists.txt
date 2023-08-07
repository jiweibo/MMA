
add_executable(gemm gemm.cu)
target_link_libraries(gemm cublas curand)
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES "75")