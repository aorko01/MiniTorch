#include"autograd/ops/MatmulGrad.h"
#include"Tensor.h"
#include"autograd/ops/Ops.h"

MatmulGrad::MatmulGrad(const Tensor& lhs, const Tensor& rhs):saved_lhs_(lhs),
saved_rhs_(rhs)
{}

std::vector<Tensor> MatmulGrad::apply(const std::vector<Tensor>& grad_outputs) 
{
    const Tensor& grad=grad_outputs[0];

    return{
        raw_mul(grad,saved_rhs_),
        raw_mul(grad,saved_lhs_)
    };
}