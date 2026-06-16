#include"autograd/ops/AddBackward.h"
#include"Tensor.h"

std::vector<Tensor> AddBackward::apply(const std::vector<Tensor>& grad_outputs) {
    return {
        grad_outputs[0],
        grad_outputs[0]
    };
}