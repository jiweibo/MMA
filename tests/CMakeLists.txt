add_executable(test_lru_cache lru_test.cc gtest_main.cc)
target_link_libraries(test_lru_cache ${DEPS})
add_test(NAME test_lru_cache COMMAND test_lru_cache)

add_executable(cute_test cute_test.cc gtest_main.cc)
target_link_libraries(cute_test ${DEPS})
add_test(NAME cute_test COMMAND cute_test)

set(CMAKE_CXX_FLAGS "-fpermissive")
add_executable(binary_test ops/binary_test.cc gtest_main.cc)
target_link_libraries(binary_test ${DEPS} binary)
add_test(NAME testbinary_test_cute COMMAND binary_test)

add_executable(coalesc_test coalesc_test.cu)
target_link_libraries(coalesc_test ${DEPS})

add_executable(simple_cute_gemm_example simple_cute_gemm_example.cu)
target_link_libraries(simple_cute_gemm_example ${DEPS} cublas)