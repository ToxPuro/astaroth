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

#include "print.h"
#include "misc.h"

static void
set_ndarray_void(const size_t element_size, const void* value, const size_t ndims, const size_t* dims, const size_t* subdims, const size_t* start, char* arr)
{
    if (ndims == 0) {
        memmove(arr, value, element_size);
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const size_t offset = prod(ndims - 1, dims);
        for (size_t i = start[ndims - 1]; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            set_ndarray_void(element_size, value, ndims - 1, dims, subdims, start, &arr[i * offset * element_size]);
    }
}

#define set_ndarray_double(value, ndims, dims, subdims, start, arr) set_ndarray_void(sizeof((value)), (double[]){as_double((value))}, (ndims), (dims), (subdims), (start), (char*)(arr))

static void test_pack_c(void)
{
    const size_t mm[]           = {8, 9};
    const size_t block_dims[]           = {2, 3};
    const size_t ndims          = ARRAY_SIZE(mm);
    const size_t count = prod(ndims, mm);

    double* buf0, *buf1, *buf2;
    ncalloc(count, buf0);
    ncalloc(count, buf1);
    ncalloc(count, buf2);
    double* buffers[] = {buf0, buf1, buf2};
    const size_t nbuffers = ARRAY_SIZE(buffers);
    
    for (size_t i = 0; i < nbuffers; ++i){
        set_ndarray_double(1 + i*4, ndims, mm, block_dims, ((size_t[]){0, 0}), buffers[i]);
        set_ndarray_double(2+ i*4, ndims, mm, block_dims, ((size_t[]){6, 0}), buffers[i]);
        set_ndarray_double(3+ i*4, ndims, mm, block_dims, ((size_t[]){0, 6}), buffers[i]);
        set_ndarray_double(4+ i*4, ndims, mm, block_dims, ((size_t[]){6, 6}), buffers[i]);
        printd_ndarray(ndims, mm, buffers[i]);
    }

    const size_t block_count = prod(ndims, block_dims);

    const size_t packlen = nbuffers * block_count;
    double* pack_buf;
    ncalloc(packlen, pack_buf);
    printd_array(packlen, pack_buf);

    pack(ndims, mm, block_dims, (size_t[]){6,6}, nbuffers, buffers, pack_buf);
    printd_array(packlen, pack_buf);

    unpack(pack_buf, ndims, mm, block_dims, (size_t[]){2,0}, nbuffers, buffers);
    for (size_t i = 0; i < nbuffers; ++i){
        printd_ndarray(ndims, mm, buffers[i]);
    }

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
