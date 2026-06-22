#pragma once

#include "autograd/GradFn.h"
#include"Tensor.h"

class MulBackward : public GradFn
{
    Tensor saved_lhs_,saved_rhs_; // must save inputs for backward
public:
    MulBackward(const Tensor &lhs, const Tensor &rhs);
    std::vector<Tensor> apply(const std::vector<Tensor> &grad_outputs) override;
};
