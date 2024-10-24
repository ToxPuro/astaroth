#include "math_utils.h"

uint64_t
to_linear(const Index& coords, const Shape& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

Index
to_spatial(const uint64_t index, const Shape& shape)
{
    Index coords(shape.count);
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}
