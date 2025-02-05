#pragma once

#include <algorithm>
#include <vector>

#include "datatypes.h"
#include "math_utils.h"
#include "type_conversion.h"

/**
 * Perform a simple decomposition of domain nn to nprocs partitions.
 * Uses a greedy algorithm to maximize the surface area to volume
 * ratio at each cut.
 */
ac::Shape decompose(const ac::Shape& nn, uint64_t nprocs);

/**
 * Perform a layered decomposition.
 * Returns a vector of decompositions ordered from higher
 * granularity (i.e. core-level) to lower (i.e. node-level).
 * The input vector nprocs_per_layer is likewise ordered from
 * high to low granularity and indicates the number of partitions
 * on each level of the decomposition hierarchy.
 *
 * For example, a process running on 8 nodes, consisting of
 * 4 GPUs, where each GPU is a multi-chip module of 2 devices,
 * the decomposition can be calculated by
 * decompose_hierarchical(nn, std::vector<uint64_t>{2, 4, 8});
 */
std::vector<ac::Shape> decompose_hierarchical(const ac::Shape&             nn,
                                              const std::vector<uint64_t>& nprocs_per_layer);

ac::Index hierarchical_to_spatial(const uint64_t                index,
                                  const std::vector<ac::Shape>& decompositions);

uint64_t hierarchical_to_linear(const ac::Index&              coords,
                                const std::vector<ac::Shape>& decompositions);

/**
 * Construct a global decomposition from hierarchical.
 * For example:
 * std::vector<ac::Shape> decompositions{ac::Shape{2, 2}, ac::Shape{4, 1}, ac::Shape{1, 4}};
 * ERRCHK((hierarchical_decomposition_to_global(decompositions) == ac::Shape{8, 8}));
 */
ac::Shape hierarchical_decomposition_to_global(const std::vector<ac::Shape>& decomposition);

void test_decomp(void);
