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

# Some toolchains inject -flto by default, which conflicts with the CUDA device
# link step used by the pybind module. Strip it so the build stays consistent.
foreach(flag_var
    CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_CUDA_FLAGS
    CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS
)
    if(DEFINED ${flag_var})
        string(REPLACE "-flto" "" ${flag_var} "${${flag_var}}")
    endif()
endforeach()

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
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(MINITORCH_CUDA_HOST_COMPILER
            NAMES g++-12 c++-12
            HINTS /usr/bin
        )

        if(MINITORCH_CUDA_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER "${MINITORCH_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
        else()
            message(FATAL_ERROR "CUDA builds require a GCC 12 host compiler. Install g++-12 or configure -DCMAKE_CUDA_HOST_COMPILER=/path/to/g++-12.")
        endif()
    endif()

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

if(USE_CUDA)
    list(APPEND MINITORCH_SOURCES src/autograd/ops/cuda_ops.cu)
endif()

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
if(USE_CUDA)
    target_link_libraries(test_tensor PRIVATE CUDA::cudart)
endif()
add_test(NAME tensor_tests COMMAND test_tensor)

# ── test_ops ─────────────────────────────────────────────────────────────────
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE minitorch)
if(USE_CUDA)
    target_link_libraries(test_ops PRIVATE CUDA::cudart)
endif()
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

pybind11_add_module(minitorch_cpp NO_EXTRAS bindings/pybind.cpp)
target_link_libraries(minitorch_cpp PRIVATE minitorch)
set_target_properties(minitorch_cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_options(minitorch_cpp PRIVATE -fno-lto)
target_link_options(minitorch_cpp PRIVATE -fno-lto)

if(USE_CUDA)
    target_link_libraries(minitorch_cpp PRIVATE CUDA::cudart)
endif()