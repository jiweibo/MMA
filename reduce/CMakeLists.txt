
add_executable(reduce_sum main.cc reduce_sum.cu)
target_link_libraries(reduce_sum ${DEPS})