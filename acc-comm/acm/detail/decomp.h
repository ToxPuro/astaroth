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
ac::shape decompose(const ac::shape& nn, uint64_t nprocs);

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
std::vector<ac::shape> decompose_hierarchical(const ac::shape&             nn,
                                              const std::vector<uint64_t>& nprocs_per_layer);

ac::index hierarchical_to_spatial(const uint64_t                index,
                                  const std::vector<ac::shape>& decompositions);

uint64_t hierarchical_to_linear(const ac::index&              coords,
                                const std::vector<ac::shape>& decompositions);

/**
 * Construct a global decomposition from hierarchical.
 * For example:
 * std::vector<ac::shape> decompositions{ac::shape{2, 2}, ac::shape{4, 1}, ac::shape{1, 4}};
 * ERRCHK((hierarchical_decomposition_to_global(decompositions) == ac::shape{8, 8}));
 */
ac::shape hierarchical_decomposition_to_global(const std::vector<ac::shape>& decomposition);

void test_decomp(void);
