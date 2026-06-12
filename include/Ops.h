#pragma once
#include"Tensor.h"

class Tensor;

Tensor add(
    const Tensor &a,
    const Tensor &b);

Tensor mul(
    const Tensor &a,
    const Tensor &b);