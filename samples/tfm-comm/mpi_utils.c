#include "mpi_utils.h"

#include <limits.h>

#include "errchk.h"
#include "math_utils.h"
#include "nalloc.h"
#include "type_conversion.h"

void
to_mpi_format(const size_t ndims, const size_t* dims, int* mpi_dims)
{
    as_int_array(ndims, dims, mpi_dims);
    reversei(ndims, mpi_dims);
}

void
to_astaroth_format(const size_t ndims, const int* mpi_dims, size_t* dims)
{
    as_size_t_array(ndims, mpi_dims, dims);
    reverse(ndims, dims);
}

int*
to_mpi_format_alloc(const size_t ndims, const size_t* arr)
{
    int* mpi_arr;
    nalloc(ndims, mpi_arr);
    to_mpi_format(ndims, arr, mpi_arr);
    return mpi_arr;
}

size_t*
to_astaroth_format_alloc(const size_t ndims, const int* mpi_arr)
{
    size_t* arr;
    nalloc(ndims, arr);
    to_astaroth_format(ndims, mpi_arr, arr);
    return arr;
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
    counter            = next_positive_integer(counter);
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
