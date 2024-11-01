#pragma once

#include "datatypes.h"

uint64_t to_linear(const Index& coords, const Shape& shape);

Index to_spatial(const uint64_t index, const Shape& shape);

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T, size_t N>
bool
within_box(const StaticArray<T, N>& coords, const StaticArray<T, N>& box_dims,
           const StaticArray<T, N>& box_offset)
{
    for (size_t i = 0; i < coords.count; ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

void test_math_utils(void);
