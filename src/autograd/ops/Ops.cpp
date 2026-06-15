#include "autograd/ops/Ops.h"
#include "utils.h"
#include <stdexcept>
// for conditional compilation
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void generic_add(const Tensor &a, const Tensor &b, Tensor &out, const std::vector<int> &shape_a)
{
    std::vector<int> idx(shape_a.size(), 0);
    int size = 1;
    for (int i = 0; i < shape_a.size(); i++)
        size *= shape_a[i];

    for (int i = 0; i < size; i++)
    {
        out.set(idx, a.get(idx) + b.get(idx));
        increment_index(idx, shape_a);
    }
}

void simd_add(const Tensor &a, const Tensor &b, Tensor &out, const std::vector<int> &shape_a)
{
    const float *tensor_a = a.get_tensor_unrolled();
    const float *tensor_b = b.get_tensor_unrolled();
    float *tensor_out = out.get_tensor_unrolled();

    int size = 1;
    for (int dim : shape_a)
        size *= dim;

    int i = 0;

    // the part of the code below would compile conditionally
    // this is for the x86 architecture
#ifdef __AVX2__

    for (; i + 7 < size; i += 8)
    {
        __m256 va = _mm256_loadu_ps(tensor_a + i);
        __m256 vb = _mm256_loadu_ps(tensor_b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(tensor_out + i, vc);
    }

    // this is for the arn architecture
#elif defined(__ARM_NEON)

    for (; i + 3 < size; i += 4)
    {
        float32x4_t va = vld1q_f32(tensor_a + i);
        float32x4_t vb = vld1q_f32(tensor_b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(tensor_out + i, vc);
    }

#endif

    for (; i < size; ++i)
        tensor_out[i] = tensor_a[i] + tensor_b[i];
}

void generic_mul(const Tensor &a, const Tensor &b, Tensor &out, const std::vector<int> &shape_a)
{
    std::vector<int> idx(shape_a.size(), 0);

    int size = 1;
    for (int i = 0; i < shape_a.size(); i++)
        size *= shape_a[i];

    for (int i = 0; i < size; i++)
    {
        out.set(idx, a.get(idx) * b.get(idx));
        increment_index(idx, shape_a);
    }
}

void simd_mul(const Tensor &a, const Tensor &b, Tensor &out, const std::vector<int> &shape_a)
{
    const float *tensor_a = a.get_tensor_unrolled();
    const float *tensor_b = b.get_tensor_unrolled();
    float *tensor_out = out.get_tensor_unrolled();

    int size = 1;
    for (int dim : shape_a)
        size *= dim;

    int i = 0;

#ifdef __AVX2__

    for (; i + 7 < size; i += 8)
    {
        __m256 va = _mm256_loadu_ps(tensor_a + i);
        __m256 vb = _mm256_loadu_ps(tensor_b + i);

        __m256 vc = _mm256_mul_ps(va, vb);

        _mm256_storeu_ps(tensor_out + i, vc);
    }

#elif defined(__ARM_NEON)

    for (; i + 3 < size; i += 4)
    {
        float32x4_t va = vld1q_f32(tensor_a + i);
        float32x4_t vb = vld1q_f32(tensor_b + i);

        float32x4_t vc = vmulq_f32(va, vb);

        vst1q_f32(tensor_out + i, vc);
    }

#endif

    for (; i < size; ++i)
        tensor_out[i] = tensor_a[i] * tensor_b[i];
}


Tensor add(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape_a = a.shape();
    std::vector<int> shape_b = b.shape();

    std::vector<int> output_shape;

    if (shape_a != shape_b)
        output_shape = broadcast_shape(shape_a, shape_b);
    else
        output_shape = shape_a;

    Tensor a_view = a.broadcast_view(output_shape);
    Tensor b_view = b.broadcast_view(output_shape);

    Tensor out(output_shape);

    if (a_view.is_contiguous() && b_view.is_contiguous())
    {
        simd_add(a_view, b_view, out, output_shape);
    }
    else
    {
        generic_add(a_view, b_view, out, output_shape);
    }

    return out;
}


Tensor mul(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape_a = a.shape();
    std::vector<int> shape_b = b.shape();

    std::vector<int> output_shape;

    if (shape_a != shape_b)
        output_shape = broadcast_shape(shape_a, shape_b);
    else
        output_shape = shape_a;

    Tensor a_view = a.broadcast_view(output_shape);
    Tensor b_view = b.broadcast_view(output_shape);

    Tensor out(output_shape);

    if (a_view.is_contiguous() && b_view.is_contiguous())
    {
        simd_mul(a_view, b_view, out, output_shape);
    }
    else
    {
        generic_mul(a_view, b_view, out, output_shape);
    }

    return out;
}
