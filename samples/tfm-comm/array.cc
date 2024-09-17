#include "array.h"

#include <stdlib.h>

// #include "math_utils.h"

real*
array_create(const size_t count, const bool on_device)
{
    return (real*)malloc(sizeof(real) * count);
}

void
array_destroy(real** array, const bool on_device)
{
    free(*array);
    *array = NULL;
}

// size_t
// array_linear_index(const size_t ndims, const size_t* shape)
// {
//     size_t* offsets = (size_t*)malloc(sizeof(offsets[0]) * ndims);
//     ERRCHK(offsets);
//     cumprod(ndims, shape, offsets);
//     rshift(1, 1, ndims, offsets);
//     free(offsets);
// }
