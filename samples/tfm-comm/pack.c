#include "pack.h"

#include "math_utils.h"
#include "misc.h"
#include "nalloc.h"
#include "print.h"

void
test_pack(void)
{
    {
        const size_t mm[]             = {8, 9};
        const size_t block_dims[]     = {2, 3};
        const size_t block_offset_a[] = {6, 6};
        const size_t block_offset_b[] = {2, 3};
        const size_t ndims            = ARRAY_SIZE(mm);
        const size_t count            = prod(ndims, mm);

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

        pack(ndims, mm, block_dims, block_offset_a, nbuffers, (double*[]){buf0, buf2, buf1},
             pack_buf);
        printd_array(packlen, pack_buf);

        unpack(pack_buf, ndims, mm, block_dims, block_offset_b, nbuffers, buffers);
        for (size_t i = 0; i < nbuffers; ++i) {
            printd_ndarray(ndims, mm, buffers[i]);
        }

        const double model_pack_buf[] = {4,  4,  4,  4, 4, 4, 12, 12, 12,
                                         12, 12, 12, 8, 8, 8, 8,  8,  8};
        ERRCHK(ncmp(packlen, pack_buf, model_pack_buf));

        const double model_buf0[] = {
            1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2,
            0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0,
            3, 3, 0, 0, 0, 0, 4, 4, 3, 3, 0, 0, 0, 0, 4, 4, 3, 3, 0, 0, 0, 0, 4, 4,
        };
        const double model_buf1[] = {
            5, 5, 0,  0,  0, 0, 6, 6, 5, 5, 0,  0,  0, 0, 6, 6, 5, 5, 0,  0,  0, 0, 6, 6,
            0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 12, 12, 0, 0, 0, 0,
            7, 7, 0,  0,  0, 0, 8, 8, 7, 7, 0,  0,  0, 0, 8, 8, 7, 7, 0,  0,  0, 0, 8, 8,
        };
        const double model_buf2[] = {
            9,  9,  0, 0, 0, 0, 10, 10, 9,  9,  0, 0, 0, 0, 10, 10, 9,  9,  0, 0, 0, 0, 10, 10,
            0,  0,  8, 8, 0, 0, 0,  0,  0,  0,  8, 8, 0, 0, 0,  0,  0,  0,  8, 8, 0, 0, 0,  0,
            11, 11, 0, 0, 0, 0, 12, 12, 11, 11, 0, 0, 0, 0, 12, 12, 11, 11, 0, 0, 0, 0, 12, 12,
        };
        ERRCHK(ncmp(count, model_buf0, buf0));
        ERRCHK(ncmp(count, model_buf1, buf1));
        ERRCHK(ncmp(count, model_buf2, buf2));

        ndealloc(pack_buf);
        ndealloc(buf0);
        ndealloc(buf1);
        ndealloc(buf2);
    }
}
