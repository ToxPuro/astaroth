#pragma once
#include <stddef.h>

size_t partition(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* nn_offset,
                 const size_t npartitions, size_t dims[npartitions][ndims],
                 size_t offsets[npartitions][ndims]);
