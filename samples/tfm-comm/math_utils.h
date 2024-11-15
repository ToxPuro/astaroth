#pragma once

#include "datatypes.h"

template <size_t N> uint64_t to_linear(const Index<N>& coords, const Shape<N>& shape);

template <size_t N> Index<N> to_spatial(const uint64_t index, const Shape<N>& shape);

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T, size_t N>
bool
within_box(const ac::array<T, N>& coords, const ac::array<T, N>& box_dims,
           const ac::array<T, N>& box_offset)
{
    for (size_t i = 0; i < coords.size(); ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

void test_math_utils(void);
