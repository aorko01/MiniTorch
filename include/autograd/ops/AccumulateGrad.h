#pragma once

#include "autograd/GradFn.h"
#include <memory>

class Tensor;

class AccumulateGrad : public GradFn
{
public:
    std::weak_ptr<Tensor> variable;

    explicit AccumulateGrad(std::shared_ptr<Tensor> var);

    std::vector<Tensor> apply(const std::vector<Tensor> &grad_outputs) override;
};