#pragma once

#include "segment.h"

#include <vector>

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
std::vector<ac::segment> partition(const Shape& mm, const Shape& nn, const Index& nn_offset);

void test_partition(void);
