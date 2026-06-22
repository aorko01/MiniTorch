#include "autograd/ops/AccumulateGrad.h"
#include "Tensor.h"
#include "autograd/ops/Ops.h"
#include<memory>


AccumulateGrad::AccumulateGrad(std::shared_ptr<Tensor> var)
    : variable(var)
{
}

std::vector<Tensor> AccumulateGrad::apply(
    const std::vector<Tensor> &grad_outputs)
{
    auto var = variable.lock();
    if (var)
    {
        
        if (!var->grad())
        {
            // First gradient
            var->set_grad(
                std::make_shared<Tensor>(grad_outputs[0]));
        }
        else
        {
            // Accumulate gradient
            var->set_grad(
                std::make_shared<Tensor>(
                    raw_add(*var->grad(), grad_outputs[0])));
        }
    }

    return {};
}