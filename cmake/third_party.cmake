include(cmake/gflags.cmake)
include(cmake/glog.cmake)
include(cmake/gtest.cmake)

add_custom_target(third_party ALL)
add_dependencies(third_party gflags glog gtest)
