#pragma once 

class Tensor;

Tensor add(
    const Tensor& a,
    const Tensor& b
);

Tensor mul(
    const Tensor& a, 
    const Tensor& b
);