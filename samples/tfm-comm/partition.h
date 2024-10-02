#pragma once
#include <stddef.h>

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
 * // Get the size for offsets and dims arrays
 * size_t nelems;
 * partition(..., &nelems, NULL, NULL);
 * size_t *offsets, *dims;
 * nalloc(nelems, offsets);
 * nalloc(nelems, dims);
 *
 * // Fill offsets and dims with the partitioning
 * const size_t npartitions = partition(..., &nelems, offsets, dims);
 *
 * // Visualize the output arranged in dims-first ordering
 * print_ndarray("offsets", 2, ((size_t[]){ndims, npartitions}), offsets));
 *
 * // Deallocate
 * dealloc(offsets);
 * dealloc(dims);
 */
size_t partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
                 size_t* nelems, size_t* dims, size_t* offsets);

void test_partition(void);
