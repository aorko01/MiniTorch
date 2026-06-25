#pragma once
#include "Tensor.h"

class Tensor;

Tensor raw_add(
    const Tensor &a,
    const Tensor &b);

Tensor raw_mul(
    const Tensor &a,
    const Tensor &b);

Tensor add(
    const Tensor &a,
    const Tensor &b);

Tensor mul(
    const Tensor &a,
    const Tensor &b);

Tensor broadcast_view(
    const Tensor &a,
    const std::vector<int> &shape);

#ifdef USE_CUDA
void cuda_raw_add_contiguous(
    const Tensor &a,
    const Tensor &b,
    Tensor &out,
    int n);

void cuda_raw_mul_contiguous(
    const Tensor &a,
    const Tensor &b,
    Tensor &out,
    int n);
#endif