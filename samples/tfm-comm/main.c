#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"
#include "ndarray.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

int
main(void)
{
    acCommTest();
    // Init
    // const size_t nn[]  = {4, 4, 4};
    // const size_t rr[]  = {1, 1, 1};
    // const size_t ndims = ARRAY_SIZE(nn);
    // acCommInit(ndims, nn, rr);

    // Compute
    // acCommHaloExchange(ndims);
    test_math_utils();
    ndarray_test();

    // Quit
    // acCommQuit();

    return EXIT_SUCCESS;
}
