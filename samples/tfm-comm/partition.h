#pragma once
#include <stddef.h>

typedef struct {
    size_t ndims;
    size_t* dims;
    size_t* offset;
} Partition;

// typedef struct {
//     size_t nrows;
//     size_t ncols;
//     size_t* data;
// } Matrix; // it's an ndarray

/** Partition the domain mm into subdomains.
 * Returns the number of partitions.
 * If `partitions` is not null, uses it to allocate a new array of partitions
 * that must be freed with partitions_destroy afterwards.
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
 */
// size_t partitions_create(const size_t ndims, const size_t* mm, const size_t* nn,
//                          const size_t* nn_offset, const size_t npartitions, Partition*
//                          partitions);

// void partitions_destroy(const size_t npartitions, Partition** partitions);

void test_partition(void);
