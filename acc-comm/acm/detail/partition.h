#pragma once

#include "segment.h"

#include <vector>

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
std::vector<ac::segment> partition(const ac::shape& mm, const ac::shape& nn,
                                   const ac::index& nn_offset);

void test_partition(void);
