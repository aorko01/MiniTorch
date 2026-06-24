#pragma once
#include <vector>
#include <string>

class Storage
{
private:
    std::vector<float> data_;
    float *cuda_data_ = nullptr;
    int size_ = 0;
    std::string device_ = "cpu";

public:
    // we're writing explicit here because we don't want silent conversions. without explicit Storage s=10 becomes valid. But we don't want that to happen
    explicit Storage(int size);
    ~Storage();

    float *data();
    const float *data() const;
    void set_device(const std::string &device);
    const std::string &device() const;
    float get_value(int index) const;
    void set_value(int index, float value);
};