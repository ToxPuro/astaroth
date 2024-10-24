#pragma once

#include "shape.h"

uint64_t to_linear(const Index& coords, const Shape& shape);

Index to_spatial(const uint64_t index, const Shape& shape);

uint64_t prod(const size_t count, const uint64_t* arr);
