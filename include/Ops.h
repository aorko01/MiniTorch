#pragma once
#include"Tensor.h"

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
    const Tensor&a ,
    const std::vector<int>& shape);