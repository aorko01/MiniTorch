#pragma once 
#include<memory>

class GradFn;

struct Edge{
    std::shared_ptr<GradFn> fn;
    int input_nr=0;
};