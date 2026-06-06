#include "Tensor.h"
#include "utils.h"
#include <stdexcept>

void Tensor::calculate_stride(std::vector<int> &shape, std::vector<int> &strides_)
{
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--)
    {
        strides_[i] = stride;
        stride *= shape[i];
    }
}

Tensor::Tensor(std::vector<int> shape)
    : shape_(shape)
{
    int size = 1;
    for (auto s : shape)
    {
        size *= s;
    }
    storage_ = std::make_shared<Storage>(size);
    strides_.resize(shape.size());

    // example for visualization
    //  (3,2,2)

    // [[[1,2],[3,4]],
    // [[1,2],[3,4]],
    // [[1,2],[3,4]]]
    calculate_stride(shape, strides_);
}

int Tensor::location(const std::vector<int> &idx) const
{
    if (idx.size() != shape_.size())
        throw std::runtime_error("bad index");

    int offset = this->offset_;
    for (int i = 0; i < idx.size(); i++)
    {
        offset += idx[i] * strides_[i];
    }
    return offset;
}

float Tensor::get(const std::vector<int> &idx) const
{
    int index = location(idx);
    // data() returns a pointer to float which is then accessed by the index properly
    return storage_->data()[index];
}

void Tensor::set(const std::vector<int> &idx, float value)
{
    int index = location(idx);
    storage_->data()[index] = value;
}

const std::vector<int>& Tensor::shape() const{
    return shape_;
}

const std::vector<int> & Tensor::strides() const{
    return strides_;
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
    Tensor out = Tensor(*this);

    if (dim0 >= shape_.size() || dim1 >= shape_.size() || dim0 < 0 || dim1 < 0)
        throw std::runtime_error("wrong dimensions");

    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

bool Tensor::is_contiguous() const
{
    int expected = 1;

    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (strides_[i] != expected)
            return false;

        expected *= shape_[i];
    }

    return true;
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
    Tensor out = Tensor(*this);
    int prev_size = 1;
    for (int i = 0; i < shape_.size(); i++)
    {
        prev_size *= shape_[i];
    }
    int new_size = 1;
    for (int i = 0; i < new_shape.size(); i++)
    {
        new_size *= new_shape[i];
    }
    if (new_size != prev_size)
        throw std::runtime_error("reshape dimension is not compatible with existing tensor");

    if (is_contiguous())
    {
        out.shape_ = new_shape;
        out.strides_.resize(new_shape.size());
        calculate_stride(out.shape_, out.strides_);
        return out;
    }

    Tensor final_(new_shape);
    int curr = 0;
    float *flattened_memory = final_.storage_->data();
    std::vector<int> idx(out.shape_.size(), 0);
    for (int i = 0; i < prev_size; i++)
    {
        flattened_memory[curr] = out.get(idx);
        curr++;
        increment_index(idx, out.shape_);
    }
    return final_;
}
