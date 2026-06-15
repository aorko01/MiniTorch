#pragma once
#include <vector>
#include <memory>
#include "Storage.h"
// #include"autograd/GradFn.h"

class GradFn;

class Tensor
{
private:
    std::shared_ptr<Storage> storage_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int offset_ = 0;

    //for autograd
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;          // accumulated gradient (same shape)
    std::shared_ptr<GradFn> grad_fn_;       // null for leaf tensors
    bool is_leaf_ = true;

    // given the indices where does the element actually lie in the 1D array
    int location(const std::vector<int> &idx) const;
    // this function would return a stride given a new shape not applicable for the reshape it should only be used given a new tensor .
    static void calculate_stride(std::vector<int> &shape, std::vector<int> &strides_);

public:
    Tensor(std::vector<int> shape);

    // getter and setter
    float get(const std::vector<int> &idx) const;
    void set(const std::vector<int> &idx, float value);
    const float *get_tensor_unrolled() const;
    float *get_tensor_unrolled();

    bool is_contiguous() const;

    const std::vector<int> &shape() const;
    const std::vector<int> &strides() const;

    // shape views and reshapes
    Tensor transpose(int dim0, int dim1) const;
    Tensor reshape(const std::vector<int> &new_shape) const;
    Tensor broadcast_view(const std::vector<int> &out_shape) const;
};