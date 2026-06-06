cmake_minimum_required(VERSION 3.10)

project(minitorch)

set(CMAKE_CXX_STANDARD 17)

# tell compiler where headers are
include_directories(include)

# build library from source files
add_library(minitorch
    src/Storage.cpp
    src/Tensor.cpp
)

enable_testing()

add_executable(test_tensor
    tests/test_tensor.cpp
)
target_link_libraries(test_tensor minitorch)
add_test(NAME tensor_tests COMMAND test_tensor)