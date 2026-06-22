#pragma once

#include "autograd/GradFn.h"

class AddBackward : public GradFn
{
    // No saved tensors needed — grad just passes through


public:
    std::vector<Tensor> apply(const std::vector<Tensor> &grad_outputs) override;
};
