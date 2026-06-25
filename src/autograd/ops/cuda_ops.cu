#include <cuda_runtime.h>
#include <stdexcept>

#include "autograd/ops/Ops.h"

namespace
{
    void check_cuda(cudaError_t status, const char *message)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(message);
        }
    }
}

__global__ void cuda_add_contiguous(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void cuda_mul_contiguous(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] * b[i];
    }
}

static void launch_add(const Tensor &a, const Tensor &b, Tensor &out, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_add_contiguous<<<blocks, threads>>>(
        const_cast<float *>(a.get_tensor_unrolled()),
        const_cast<float *>(b.get_tensor_unrolled()),
        out.get_tensor_unrolled(),
        n);
    check_cuda(cudaGetLastError(), "CUDA add kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "CUDA add kernel execution failed");
}

static void launch_mul(const Tensor &a, const Tensor &b, Tensor &out, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_mul_contiguous<<<blocks, threads>>>(
        const_cast<float *>(a.get_tensor_unrolled()),
        const_cast<float *>(b.get_tensor_unrolled()),
        out.get_tensor_unrolled(),
        n);
    check_cuda(cudaGetLastError(), "CUDA mul kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "CUDA mul kernel execution failed");
}

void cuda_raw_add_contiguous(const Tensor &a, const Tensor &b, Tensor &out, int n)
{
    launch_add(a, b, out, n);
}

void cuda_raw_mul_contiguous(const Tensor &a, const Tensor &b, Tensor &out, int n)
{
    launch_mul(a, b, out, n);
}