# MiniTorch

## Build

Create a separate build directory and configure with CMake.

### CPU-only build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### CUDA build with nvcc

Use this when CUDA is installed and `nvcc` is available at the path below.

```bash
mkdir -p build
cd build
cmake .. \
	-DUSE_CUDA=ON \
	-DCMAKE_CUDA_COMPILER=/usr/lib/nvidia-cuda-toolkit/bin/nvcc \
	-DCMAKE_PREFIX_PATH=/usr/lib/nvidia-cuda-toolkit
cmake --build .
```

### Run tests

```bash
ctest
```
