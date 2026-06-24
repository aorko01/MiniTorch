# cmake_minimum_required(VERSION 3.10)

# project(minitorch)

# set(CMAKE_CXX_STANDARD 17)

# # tell compiler where headers are
# include_directories(include)

# # build library from source files
# add_library(minitorch
#     src/Storage.cpp
#     src/Tensor.cpp
#     src/autograd/ops/Ops.cpp
#     src/autograd/ops/AddBackward.cpp
#     src/autograd/ops/MulBackward.cpp
#     src/autograd/ops/AccumulateGrad.cpp
#     src/autograd/Engine.cpp
# )
# set_target_properties(minitorch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# enable_testing()

# add_executable(test_tensor
#     tests/test_tensor.cpp
#     # tests/test_ops.cpp
# )
# target_link_libraries(test_tensor minitorch)
# add_test(NAME tensor_tests COMMAND test_tensor)

# add_executable(test_ops
#     tests/test_ops.cpp
# )
# target_link_libraries(test_ops minitorch)
# add_test(NAME ops_tests COMMAND test_ops)

# find_package(pybind11 REQUIRED)

# pybind11_add_module(
#     minitorch_cpp
#     bindings/pybind.cpp
# )

# target_link_libraries(
#     minitorch_cpp
#     PRIVATE
#     minitorch
# )


cmake_minimum_required(VERSION 3.18)  # 3.18+ required for CUDAToolkit + native CUDA language support

project(minitorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
#  Optional CUDA support
#  Enabled automatically when nvcc is found, or force with -DUSE_CUDA=ON/OFF.
# ─────────────────────────────────────────────────────────────────────────────

option(USE_CUDA "Build with CUDA support" AUTO)

if(USE_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(USE_CUDA ON)
    else()
        set(USE_CUDA OFF)
    endif()
endif()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Default to sm_75 (Turing) if the user hasn't set anything.
    # Override on the command line: -DCMAKE_CUDA_ARCHITECTURES="80;86"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    message(STATUS "CUDA enabled — compiler: ${CMAKE_CUDA_COMPILER}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    add_compile_definitions(USE_CUDA)
else()
    message(STATUS "CUDA not found — building CPU-only. GPU tests will be skipped.")
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Include paths
# ─────────────────────────────────────────────────────────────────────────────

include_directories(include)

# ─────────────────────────────────────────────────────────────────────────────
#  Core library
# ─────────────────────────────────────────────────────────────────────────────

set(MINITORCH_SOURCES
    src/Storage.cpp
    src/Tensor.cpp
    src/autograd/ops/Ops.cpp
    src/autograd/ops/AddBackward.cpp
    src/autograd/ops/MulBackward.cpp
    src/autograd/ops/AccumulateGrad.cpp
    src/autograd/Engine.cpp
)

add_library(minitorch ${MINITORCH_SOURCES})
set_target_properties(minitorch PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(USE_CUDA)
    # Allow .cpp files in the library to see CUDA headers.
    target_include_directories(minitorch PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(minitorch PRIVATE CUDA::cudart)

    # Separable compilation lets device code call functions across .cu files.
    set_target_properties(minitorch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

enable_testing()

# ── test_tensor (CPU-only, always built) ─────────────────────────────────────
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE minitorch)
add_test(NAME tensor_tests COMMAND test_tensor)

# ── test_ops ─────────────────────────────────────────────────────────────────
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE minitorch)
add_test(NAME ops_tests COMMAND test_ops)

# ── test_device (only built when CUDA is available) ───────────────────────────
if(USE_CUDA)
    add_executable(test_device tests/test_device.cpp)
    target_link_libraries(test_device PRIVATE minitorch CUDA::cudart)
    add_test(NAME device_tests COMMAND test_device)

    # Propagate the CUDA include path so test_device.cpp can #include <cuda_runtime.h>
    target_include_directories(test_device PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Label the test so you can run only GPU tests with: ctest -L gpu
    set_tests_properties(device_tests PROPERTIES LABELS "gpu")
endif()

# ─────────────────────────────────────────────────────────────────────────────
#  Python bindings (pybind11)
# ─────────────────────────────────────────────────────────────────────────────

find_package(pybind11 REQUIRED)

pybind11_add_module(minitorch_cpp bindings/pybind.cpp)
target_link_libraries(minitorch_cpp PRIVATE minitorch)

if(USE_CUDA)
    target_link_libraries(minitorch_cpp PRIVATE CUDA::cudart)
endif()