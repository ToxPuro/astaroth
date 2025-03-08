#pragma once

#include <vector>

#include "segment.h"

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
std::vector<ac::segment> partition(const ac::shape& mm, const ac::shape& nn,
                                   const ac::index& nn_offset);
