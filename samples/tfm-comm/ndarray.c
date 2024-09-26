#include "ndarray.h"

#include "array.h"
#include "errchk.h"
#include "math_utils.h"
#include "print.h"

void
set_ndarray(const size_t value, const size_t ndims, const size_t* start, const size_t* subdims,
            const size_t* dims, size_t* arr)
{
    if (ndims == 0) {
        *arr = value;
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const size_t offset = prod(ndims - 1, dims);
        for (size_t i = start[ndims - 1]; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            set_ndarray(value, ndims - 1, start, subdims, dims, &arr[i * offset]);
    }
}

static size_t
nd_to_1d(const size_t ndims, const size_t* coords, const size_t* dims)
{
    ERRCHK(all_less_than(ndims, coords, dims));
    size_t offset[ndims];
    cumprod(ndims, dims, offset);
    rshift(1, 1, ndims, offset);
    return dot(ndims, coords, offset);
}

static void
test_nd_to_1d(void)
{
    {
        const size_t coords[] = {0, 0, 0};
        const size_t dims[]   = {1, 1, 1};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 0);
    }
    {
        const size_t coords[] = {1, 0};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 1);
    }
    {
        const size_t coords[] = {31, 0};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 31);
    }
    {
        const size_t coords[] = {0, 31};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 31 * 32);
    }
    {
        const size_t coords[] = {1, 2, 3, 4};
        const size_t dims[]   = {10, 9, 8, 7};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 1 + 2 * 10 + 3 * 10 * 9 + 4 * 10 * 9 * 8);
    }
}

bool
ndarray_equals(const size_t count, const size_t ndims, const size_t* a_offset,
               const size_t* b_offset, const size_t* dims, const size_t* arr)
{
    const size_t a = nd_to_1d(ndims, a_offset, dims);
    const size_t b = nd_to_1d(ndims, b_offset, dims);
    return equals(count, &arr[a], &arr[b]);
}

void
test_ndarray_equals(void)
{
    const size_t arr[] = {
        1, 1, 1, //
        1, 2, 3, //
        1, 1, 1, //
        3, 2, 1, //
    };
    const size_t ncols  = 3;
    const size_t len    = ARRAY_SIZE(arr);
    const size_t nrows  = len / ncols;
    const size_t dims[] = {ncols, nrows};
    const size_t ndims  = ARRAY_SIZE(dims);
    {
        const size_t a_offset[] = {0, 0};
        const size_t b_offset[] = {0, 1};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == false);
    }
    {
        const size_t a_offset[] = {0, 0};
        const size_t b_offset[] = {0, 2};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == true);
    }
    {
        const size_t a_offset[] = {0, 1};
        const size_t b_offset[] = {0, 3};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == false);
    }
    {
        const size_t a_offset[] = {2, 2};
        const size_t b_offset[] = {2, 3};
        ERRCHK(ndarray_equals(1, ndims, a_offset, b_offset, dims, arr) == true);
    }
}

static void
print_ndarray_recursive(const size_t ndims, const size_t* dims, const size_t* arr)
{
    if (ndims == 1) {
        for (size_t i = 0; i < dims[0]; ++i) {
            const size_t len          = 128;
            const int print_alignment = 3;
            char str[len];
            snprintf(str, len, format_specifier(arr[i]), arr[i]);
            printf("%*s ", print_alignment, str);
        }
        printf("\n");
    }
    else {
        const size_t offset = prod(ndims - 1, dims);
        for (size_t i = 0; i < dims[ndims - 1]; ++i) {
            if (ndims > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);
            if (ndims == 4)
                printf("Cube %zu:\n", i);
            if (ndims == 3)
                printf("Layer %zu:\n", i);
            if (ndims == 2)
                printf("Row %zu:", i);
            print_ndarray_recursive(ndims - 1, dims, &arr[i * offset]);
        }
        printf("\n");
    }
}

void
print_ndarray(const char* label, const size_t ndims, const size_t* dims, const size_t* arr)
{
    printf("%s:\n", label);
    print_ndarray_recursive(ndims, dims, arr);
}

void
test_ndarray(void)
{
    test_nd_to_1d();
    test_ndarray_equals();
}
