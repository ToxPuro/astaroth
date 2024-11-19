#pragma once

#include "datatypes.h"

template <size_t N>
uint64_t
to_linear(const ac::array<uint64_t, N>& coords, const ac::array<uint64_t, N>& shape)
{
    uint64_t result{0};
    for (size_t j{0}; j < shape.size(); ++j) {
        uint64_t factor{1};
        for (size_t i{0}; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <size_t N>
Index<N>
to_spatial(const uint64_t index, const ac::array<uint64_t, N>& shape)
{
    ac::array<uint64_t, N> coords;
    for (size_t j{0}; j < shape.size(); ++j) {
        uint64_t divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T, size_t N>
bool
within_box(const ac::array<T, N>& coords, const ac::array<T, N>& box_dims,
           const ac::array<T, N>& box_offset)
{
    for (size_t i{0}; i < coords.size(); ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

void test_math_utils(void);
