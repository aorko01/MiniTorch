#include "Storage.h"

// direct initialization that prevents a copy in the constructor
Storage::Storage(int size)
    : data_(size)
{
}

float *Storage::data()
{
    // data_.data returns the float pointer not the whole vector container that has other things like size and other member function. In short this is faster and cheaper
    return data_.data();
}