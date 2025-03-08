#include "math_utils.h"

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res{1};
    for (size_t i{0}; i < count; ++i)
        res *= arr[i];
    return res;
}
