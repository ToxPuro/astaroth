#pragma once

#include "datatypes.h"

uint64_t to_linear(const ac::vector<uint64_t>& coords, const ac::vector<uint64_t>& shape);

Index to_spatial(const uint64_t index, const ac::vector<uint64_t>& shape);

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T>
bool
within_box(const ac::vector<T>& coords, const ac::vector<T>& box_dims,
           const ac::vector<T>& box_offset)
{
    for (size_t i{0}; i < coords.size(); ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

void test_math_utils(void);
