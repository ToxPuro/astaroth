#pragma once

#include "shape.h"

#include <vector>

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
std::vector<Segment> partition(const size_t ndims, const Shape& mm, const Shape& nn,
                               const Index& nn_offset);

/**
 * Decompose the
 */
