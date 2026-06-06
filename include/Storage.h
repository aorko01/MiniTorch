#pragma once
#include <vector>

class Storage
{
private:
    std::vector<float> data_;

public:
    // we're writing explicit here because we don't want silent conversions. without explicit Storage s=10 becomes valid. But we don't want that to happen
    explicit Storage(int size);

    float *data();
};