#pragma once

#include <vector>

#include "datatypes.h"

/**
 * Perform a simple decomposition of domain nn to nprocs partitions.
 * Uses a greedy algorithm to maximize the surface area to volume
 * ratio at each cut.
 */
template <size_t N> Shape<N> decompose(const Shape<N>& nn, uint64_t nprocs);

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
template <size_t N>
std::vector<Shape<N>> decompose_hierarchical(const Shape<N>& nn,
                                             const std::vector<uint64_t>& nprocs_per_layer);

void test_decomp(void);
