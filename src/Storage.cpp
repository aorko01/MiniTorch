#include "Storage.h"
#include <stdexcept>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define MINITORCH_HAS_CUDA_RUNTIME 1
#include <cuda_runtime.h>
#else
#define MINITORCH_HAS_CUDA_RUNTIME 0
#endif
#else
#define MINITORCH_HAS_CUDA_RUNTIME 0
#endif

// direct initialization that prevents a copy in the constructor
Storage::Storage(int size)
    : data_(size), size_(size)
{
}

Storage::~Storage()
{
    //if device doesn't have cuda then the destructor doesn't see the code below 
#if MINITORCH_HAS_CUDA_RUNTIME
    if (cuda_data_ != nullptr)
    {
        cudaFree(cuda_data_);
        cuda_data_ = nullptr;
    }
#endif
}

float *Storage::data()
{
    if (device_ == "cuda")
    {
        return cuda_data_;
    }
    return data_.data();
}

const float *Storage::data() const
{
    if (device_ == "cuda")
    {
        return cuda_data_;
    }
    return data_.data();
}

void Storage::set_device(const std::string &device)
{
    if (device != "cpu" && device != "cuda")
    {
        throw std::runtime_error("device must be either 'cpu' or 'cuda'");
    }

    if (device_ == device)
    {
        return;
    }

    if (device == "cuda")
    {
#if MINITORCH_HAS_CUDA_RUNTIME
        if (cuda_data_ == nullptr)
        {
            //if cuda-data is nullptr then it points to no valid address. 
            //suppose we want to allocate 1024 bytes of float in the gpu then we would tell cudaMalloc to do that 
            // cuda malloc would  allocate if that amount of memory if available then it would return the address of the allocated memory to the host . 
            // now host saves the address of GPU in the main memory cuda_data_. so that is a float** (pointer to pointer) 
            //but cudaMalloc expects a void pointer that's why we cast the cuda_malloc to void pointer.
            cudaError_t alloc_status = cudaMalloc(reinterpret_cast<void **>(&cuda_data_), sizeof(float) * size_);
            if (alloc_status != cudaSuccess)
            {
                throw std::runtime_error("cudaMalloc failed while moving storage to CUDA");
            }
        }

        cudaError_t copy_status = cudaMemcpy(cuda_data_, data_.data(), sizeof(float) * size_, cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy host-to-device failed while moving storage to CUDA");
        }

        device_ = "cuda";
#else
        throw std::runtime_error("CUDA runtime is not available in this build");
#endif
        return;
    }

    if (device == "cpu")
    {
#if MINITORCH_HAS_CUDA_RUNTIME
        if (cuda_data_ != nullptr)
        {
            cudaError_t copy_status = cudaMemcpy(data_.data(), cuda_data_, sizeof(float) * size_, cudaMemcpyDeviceToHost);
            if (copy_status != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpy device-to-host failed while moving storage to CPU");
            }

            cudaFree(cuda_data_);
            cuda_data_ = nullptr;
        }
#endif
        device_ = "cpu";
    }
}

const std::string &Storage::device() const
{
    return device_;
}

float Storage::get_value(int index) const
{
    if (index < 0 || index >= size_)
    {
        throw std::runtime_error("storage index out of bounds");
    }

    if (device_ == "cpu")
    {
        return data_[index];
    }

#if MINITORCH_HAS_CUDA_RUNTIME
    float value = 0.0f;
    //first copy the value to the main memory from cuda memory and then return 
    cudaError_t copy_status = cudaMemcpy(&value, cuda_data_ + index, sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpy device-to-host failed in get_value");
    }
    return value;
#else
    throw std::runtime_error("CUDA runtime is not available in this build");
#endif
}

void Storage::set_value(int index, float value)
{
    if (index < 0 || index >= size_)
    {
        throw std::runtime_error("storage index out of bounds");
    }

    if (device_ == "cpu")
    {
        data_[index] = value;
        return;
    }

#if MINITORCH_HAS_CUDA_RUNTIME
    cudaError_t copy_status = cudaMemcpy(cuda_data_ + index, &value, sizeof(float), cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpy host-to-device failed in set_value");
    }
#else
    throw std::runtime_error("CUDA runtime is not available in this build");
#endif
}