#include "mpi_utils.h"

#include <limits.h>

#include "alloc.h"
#include "errchk.h"
#include "type_conversion.h"

void
to_mpi_format(const size_t ndims, const uint64_t* dims, int* mpi_dims)
{
    for (size_t i = 0; i < ndims; ++i)
        mpi_dims[i] = as_int(dims[i]);
    ac_reverse(ndims, sizeof(mpi_dims[0]), mpi_dims);
}

void
to_astaroth_format(const size_t ndims, const int* mpi_dims, uint64_t* dims)
{
    for (size_t i = 0; i < ndims; ++i)
        dims[i] = as_uint64_t(mpi_dims[i]);
    ac_reverse(ndims, sizeof(dims[0]), dims);
}

/**
 * At each call, returns the next integer in range [0, INT_MAX].
 * If the counter overflows, the counter wraps around and starts again from 0.
 * NOTE: Not thread-safe.
 */
int
get_tag(void)
{
    static int counter = -1;

    ++counter;
    if (counter < 0)
        counter = 0;

    return counter;
}

static void
test_get_tag(void)
{
    {
        int prev = get_tag();
        for (size_t i = 0; i < 1000; ++i) {
            int curr = get_tag();
            if (prev == INT_MAX)
                ERRCHK(curr == 0);
            else
                ERRCHK(curr > prev);
            prev = curr;
        }
    }
}

void
test_mpi_utils(void)
{
    test_get_tag();
}
