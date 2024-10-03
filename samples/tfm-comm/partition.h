#pragma once
#include <stddef.h>

#include "dynarr.h"
#include "segment.h"

typedef dynarr_s(Segment) SegmentArray;

typedef dynarr_s(size_t) DynamicArray;

/** Partitions the domain mm into subdomains divided by nn
 *  Returns the number of partitions
 * offsets and dims must be large enough to hold nelems elements
 *
 *
 * The domain mm is partitioned three subdomains in each dimension,
 * where the coordinates in dimension i are constructed from the combinations of
 * [a, b),
 * [b, c), and
 * [c, d), where
 *
 * a: 0
 * b: nn_offset[i]
 * c: nn_offset[i] + nn[i]
 * d: mm[i]
 *
 * |---|--------|---|
 * a   b        c   d
 *
 *
 * Usage:
 *
 * // Create dynamic arrays for the partitioning
 * DynamicArray segment_dims, segment_offsets;
 * dynarr_create(&segment_dims);
 * dynarr_create(&segment_offsets);
 *
 * // Fill offsets and dims with the partitioning
 * const size_t npartitions = partition(..., segment_dims, segment_offsets);
 *
 * // Visualize the output arranged in dims-first ordering
 * print_ndarray("offsets", 2, ((size_t[]){ndims, npartitions}), offsets.data));
 *
 * // Deallocate
 * dynarr_destroy(&offsets);
 * dynarr_destroy(&dims);
 */
size_t partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
                 DynamicArray* segment_dims, DynamicArray* segment_offsets);

void partition_new(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
                   SegmentArray* segments);

void test_partition(void);
