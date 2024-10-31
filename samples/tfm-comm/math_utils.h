#pragma once

#include "datatypes.h"

uint64_t to_linear(const Index& coords, const Shape& shape);

Index to_spatial(const uint64_t index, const Shape& shape);

uint64_t prod(const size_t count, const uint64_t* arr);

bool within_box(const Index& coords, const Shape& box_dims, const Index& box_offset);

void test_math_utils(void);
