#pragma once
// #include <algorithm>
#include <vector>

inline void increment_index(
    std::vector<int> &idx,
    const std::vector<int> &shape)
{
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        idx[i]++;
        if (idx[i] < shape[i])
            return;
        idx[i] = 0;
    }
}
inline std::vector<int> broadcast_shape(
    const std::vector<int> &a,
    const std::vector<int> &b)
{
    int n = std::max(a.size(), b.size());

    std::vector<int> out(n);

    for (int i = 0; i < n; i++)
    {
        int ai = i - (n - (int)a.size());
        int bi = i - (n - (int)b.size());

        int adim = (ai >= 0) ? a[ai] : 1;
        int bdim = (bi >= 0) ? b[bi] : 1;

        if (adim == bdim)
        {
            out[i] = adim;
        }
        else if (adim == 1)
        {
            out[i] = bdim;
        }
        else if (bdim == 1)
        {
            out[i] = adim;
        }
        else
        {
            throw std::runtime_error("Broadcasting error: incompatible shapes");
        }
    }

    return out;
}

