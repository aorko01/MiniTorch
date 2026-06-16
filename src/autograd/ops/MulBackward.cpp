#include"autograd/ops/MulBackward.h"
#include"Tensor.h"
#include"autograd/ops/Ops.h"

MulBackward::MulBackward(const Tensor& lhs, const Tensor& rhs):saved_lhs_(lhs),
saved_rhs_(rhs)
{}


std::vector<Tensor> MulBackward::apply(const std::vector<Tensor>& grad_outputs) 
{
    const Tensor& grad=grad_outputs[0];

    return{
        raw_mul(grad,saved_rhs_),
        raw_mul(grad,saved_lhs_)
    };
}