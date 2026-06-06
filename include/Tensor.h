#pragma once
#include <vector>
#include <memory>
#include "Storage.h"

class TensorImpl;

class Tensor
{
private:
    std::shared_ptr<Storage> storage_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int offset_ = 0;

    // given the indices where does the element actually lie in the 1D array
    int location(const std::vector<int> &idx) const;
    bool is_contiguous() const;
    //this function would return a stride given a new shape not applicable for the reshape it should only be used given a new tensor .
    static void calculate_stride(std::vector<int>& shape,std::vector<int>& strides_) ;

public:
    Tensor(std::vector<int> shape);

    //getter and setter
    float get(const std::vector<int> &idx) const;
    void set(const std::vector<int> &idx, float value);

    const std::vector<int> &shape() const;
    const std::vector<int> &strides() const;

    // shape views and reshapes
    Tensor transpose(int dim0, int dim1) const;
    Tensor reshape(const std::vector<int>& new_shape) const;

};