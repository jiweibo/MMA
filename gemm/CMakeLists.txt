# add_executable(gemm gemm.cu)
# target_link_libraries(gemm cublas curand)

# add_library(batch_gemm cutlass_batch_gemm.cu)
# set_target_properties(batch_gemm PROPERTIES CUDA_ARCHITECTURES "75")
# target_include_directories(batch_gemm PUBLIC ${CUTLASS_DIR}/include)

add_library(cutlass_gemm cutlass_gemm.cu)

#cuda_add_executable(main main.cc)
add_executable(main main.cc)
target_link_libraries(main glog cudart cutlass_gemm cublas cublasLt ${DEPS})