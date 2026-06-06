#pragma once
#include <vector>

inline void increment_index(std::vector<int> &idx, const std::vector<int> &shape)
{
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        idx[i]++;
        if (idx[i] < shape[i])
            return;
        idx[i] = 0;
    }
}