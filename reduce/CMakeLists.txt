
add_executable(reduce_sum main.cc reduce_sum.cu)
# target_link_libraries(gemm cublas curand)
set_target_properties(reduce_sum PROPERTIES CUDA_ARCHITECTURES "75")