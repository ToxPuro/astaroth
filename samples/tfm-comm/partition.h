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
 * // The array must be constructed with a destructor
 * // Otherwise the caller must manually destroy the elements afterwards
 * SegmentArray segments;
 * dynarr_create_with_destructor(segment_destroy, &segment_dims);
 *
 * // Fill offsets and dims with the partitioning
 * partition(..., segments);
 *
 * // Visualize the output arranged in dims-first ordering
 * for (size_t i = 0; i < segments.length; ++i)
 *      print_segment("-", segments.data[i]);
 *
 * // Deallocate
 * dynarr_destroy(&segments);
 */
void partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
               SegmentArray* segments);

void test_partition(void);
