#include "autograd/Engine.h"
#include "Tensor.h"
#include "autograd/GradFn.h"
#include <unordered_set>
#include <algorithm>

static void dfs(
    std::shared_ptr<GradFn> fn,
    std::vector<std::shared_ptr<GradFn>>& order,
    std::unordered_set<GradFn*>& visited)
{
    if (!fn || visited.count(fn.get()))
        return;
    visited.insert(fn.get());
    for (const Edge& e : fn->next_func)
        dfs(e.fn, order, visited);
    order.push_back(fn);
}

std::vector<std::shared_ptr<GradFn>>
Engine::topological_sort(std::shared_ptr<GradFn> root_fn)
{
    std::vector<std::shared_ptr<GradFn>> order;
    std::unordered_set<GradFn*> visited;
    dfs(root_fn, order, visited);
    return order;
}

void Engine::backward(Tensor& root, const Tensor& grad_output)
{
    auto root_fn=root.grad_fn();
    if(!root_fn)
        throw std::runtime_error("backward called on tensor with no grad function ");
    auto order = topological_sort(root_fn);
    std::reverse(order.begin(), order.end());//because we pushed in the vector and it was in reverse order 
    // S
    // t
    // u 
    // c 
    // k 

    // at 

    // D.
    // P

    // need to study dp and then come back 
}