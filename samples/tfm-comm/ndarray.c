#include "ndarray.h"

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

void
print_ndarray(const size_t ndims, const size_t* dims, const size_t* arr)
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
            print_ndarray(ndims - 1, dims, &arr[i * offset]);
        }
        printf("\n");
    }
}
