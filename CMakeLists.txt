cmake_minimum_required(VERSION 3.10)

project(minitorch)

set(CMAKE_CXX_STANDARD 17)

# tell compiler where headers are
include_directories(include)

# build library from source files
add_library(minitorch
    src/Storage.cpp
    src/Tensor.cpp
    src/Ops.cpp
)
set_target_properties(minitorch PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_executable(test_tensor
    tests/test_tensor.cpp
    # tests/test_ops.cpp
)
target_link_libraries(test_tensor minitorch)
add_test(NAME tensor_tests COMMAND test_tensor)

add_executable(test_ops
    tests/test_ops.cpp
)
target_link_libraries(test_ops minitorch)
add_test(NAME ops_tests COMMAND test_ops)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    minitorch_cpp
    bindings/pybind.cpp
)

target_link_libraries(
    minitorch_cpp
    PRIVATE
    minitorch
)