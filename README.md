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

### CPU-only build

```bash
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build
```

### CUDA build with nvcc

Use this when CUDA is installed and `nvcc` is available. If `nvcc` is not on
your PATH, point CMake at it explicitly.

```bash
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build
```

If your system needs an explicit CUDA path, use:

```bash
cmake -S . -B build \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/lib/nvidia-cuda-toolkit/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/lib/nvidia-cuda-toolkit
cmake --build build
```

### Run tests

```bash
cd build
ctest --output-on-failure
```
