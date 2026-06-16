#include<iostream>
#include<vector>

class GradFn;
class Tensor;

class Engine{
    private:
        static std::vector<std::shared_ptr<GradFn>> topological_sort(std::shared_ptr<GradFn> root_fn);
    public:
        static void backward(Tensor &root, const Tensor& grad_output);
};