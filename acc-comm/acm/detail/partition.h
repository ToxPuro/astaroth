#pragma once

#include "segment.h"

#include <vector>

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
std::vector<ac::Segment> partition(const ac::Shape& mm, const ac::Shape& nn,
                                   const ac::Index& nn_offset);

void test_partition(void);
