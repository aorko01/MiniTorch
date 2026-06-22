#pragma once

#include <vector>
#include<memory>
#include"Edge.h"

class Tensor;
class GradFn
{
public:
    std::vector<Edge> next_func;

    virtual std::vector<Tensor> apply(const std::vector<Tensor> &grad_outputs) = 0;

    // without this virtual destructor the derived class objects might not get destroyed leading to memory leak
    virtual ~GradFn() = default;
};