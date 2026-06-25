# MiniTorch

## Build

Create a separate build directory and configure with CMake from the project
root.

```bash
cmake -S . -B build
cmake --build build
```

By default, the project tries to detect CUDA automatically. If you want to be
explicit, use one of the options below.

### Prerequisites

To reproduce the CUDA build on this setup, make sure these tools are installed:

```bash
sudo apt install cmake g++-12 nvidia-cuda-toolkit
```

If your CUDA installation lives somewhere else, adjust the compiler and toolkit
paths in the commands below.

### CPU-only build

```bash
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build
```

### CUDA build with nvcc

Use this when CUDA is installed and `nvcc` is available. This toolchain also
needs a supported host compiler; on this setup that means GCC/G++ 12.

```bash
cmake -S . -B build \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build
```

If `nvcc` is not on your PATH, or CUDA lives in a non-default location, use:

```bash
cmake -S . -B build \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/lib/nvidia-cuda-toolkit/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/lib/nvidia-cuda-toolkit \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build
```

If you do not have GCC/G++ 12 installed yet, install it first and use that
compiler for the CUDA configure step.

### Run all tests

```bash
cd build
ctest --output-on-failure
```

`ctest` runs the full test suite that was generated for the current build. If
you configured CUDA support, this includes the CUDA-backed tests as well.
