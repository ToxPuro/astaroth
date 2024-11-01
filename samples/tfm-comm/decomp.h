#pragma once

#include "datatypes.h"

/**
 * Perform a simple decomposition of domain nn to nprocs partitions.
 * Uses a greedy algorithm to maximize the surface area to volume
 * ratio at each cut.
 */
Shape decompose(const Shape& nn, uint64_t nprocs);

/**
 * Perform a layered decomposition.
 * Returns a vector of decompositions ordered from lower
 * granularity (i.e. node-level) to higher (i.e. core-level).
 * The input vector nprocs_per_layer is likewise ordered from
 * low to high granularity and indicates the number of partitions
 * on each level of the decomposition hierarchy.
 *
 * For example, a process running on 8 nodes, consisting of
 * 4 GPUs, where each GPU is a multi-chip module of 2 devices,
 * the decomposition can be calculated by
 * decompose_hierarchical(nn, std::vector<uint64_t>{8, 4, 2});
 */
std::vector<Shape> decompose_hierarchical(const Shape& nn, std::vector<uint64_t>& nprocs_per_layer);

void test_decomp(void);
