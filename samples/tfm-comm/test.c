#include "test.h"

#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "print.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

int
main(void)
{
    printf("Hello from test\n");

    {
        const size_t arr[] = {1, 2, 3, 4, 5};
        const size_t count = ARRAY_SIZE(arr);
        print_array("Input", count, arr);

        size_t out[count];
        rshift(1, 1, count, arr, out);
        print_array("Output", count, out);

        cumprod(count, arr, out);
        print_array("Output", count, out);
    }

    {
        const size_t dims[] = {128, 128, 128};
        const size_t ndims  = ARRAY_SIZE(dims);
        size_t offsets[ndims];
        rshift(1, 1, ndims, dims, offsets);
        cumprod(ndims, offsets, offsets);
        print_array("Offsets", ndims, offsets);

        const size_t pos[] = {1, 1, 1};
        dot(ndims, pos, offsets);
        print_array("Position", ndims, pos);
        print("Index", dot(ndims, pos, offsets));
    }

    {
        size_t arr[]       = {1, 2, 3, 3, 4, 5};
        const size_t count = ARRAY_SIZE(arr);
        print("Unique", unique2(count, arr));
    }

    return EXIT_SUCCESS;
}
