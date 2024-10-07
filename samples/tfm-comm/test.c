#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "dynarr.h"
#include "math_utils.h"
#include "mpi_utils.h"
#include "pack.h"
#include "partition.h"
#include "segment.h"
#include "type_conversion.h"

#include "misc.h"
#include "print.h"

// #define set_ndarray_double(value, ndims, dims, subdims, start, arr)                                \
//     set_ndarray(sizeof((value)), (double[]){as_double((value))}, (ndims), (dims), (subdims),       \
//                 (start), (char*)(arr))

// static void
// set_ndarray_double(const double value, const size_t ndims, const size_t* dims,
//                    const size_t* subdims, const size_t* start, double* arr)
// {
//     set_ndarray_void(sizeof(value), (double[]){value}, ndims, dims, subdims, start, (char*)arr);
// }

// #define set_ndarray(value, ndims, dims, subdims, start, arr)                                       \
//     _Generic((arr[0]), double: set_ndarray_double)(value, ndims, dims, subdims, start, arr)

static void
test_pack_c(void)
{
    const size_t mm[]         = {8, 9};
    const size_t block_dims[] = {2, 3};
    const size_t ndims        = ARRAY_SIZE(mm);
    const size_t count        = prod(ndims, mm);

    double *buf0, *buf1, *buf2;
    ncalloc(count, buf0);
    ncalloc(count, buf1);
    ncalloc(count, buf2);
    double* buffers[]     = {buf0, buf1, buf2};
    const size_t nbuffers = ARRAY_SIZE(buffers);

    for (size_t i = 0; i < nbuffers; ++i) {
        set_ndarray_double(1 + i * 4, ndims, mm, block_dims, ((size_t[]){0, 0}), buffers[i]);
        set_ndarray_double(2 + i * 4, ndims, mm, block_dims, ((size_t[]){6, 0}), buffers[i]);
        set_ndarray_double(3 + i * 4, ndims, mm, block_dims, ((size_t[]){0, 6}), buffers[i]);
        set_ndarray_double(4 + i * 4, ndims, mm, block_dims, ((size_t[]){6, 6}), buffers[i]);
        printd_ndarray(ndims, mm, buffers[i]);
    }

    const size_t block_count = prod(ndims, block_dims);

    const size_t packlen = nbuffers * block_count;
    double* pack_buf;
    ncalloc(packlen, pack_buf);
    printd_array(packlen, pack_buf);

    pack(ndims, mm, block_dims, (size_t[]){6, 6}, nbuffers, buffers, pack_buf);
    printd_array(packlen, pack_buf);

    unpack(pack_buf, ndims, mm, block_dims, (size_t[]){2, 0}, nbuffers, buffers);
    for (size_t i = 0; i < nbuffers; ++i) {
        printd_ndarray(ndims, mm, buffers[i]);
    }

    ndealloc(pack_buf);
    ndealloc(buf0);
    ndealloc(buf1);
    ndealloc(buf2);
}

int
main(void)
{
    test_comm();
    test_dynarr();
    test_math_utils();
    test_mpi_utils();
    test_pack();
    test_partition();
    test_segment();

    test_pack_c();
    return EXIT_SUCCESS;
}
