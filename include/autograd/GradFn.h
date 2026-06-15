#include <vector>
#include "Tensor.h"
class GradFn
{
public:
    std::vector<std::shared_ptr<GradFn>> next_func;

    virtual std::vector<Tensor> apply(const std::vector<Tensor> &grad_outputs) = 0;

    //without this virtual destructor the derived class objects might not get destroyed leading to memory leak 
    virtual ~GradFn() = default;

};