#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"
#include "ndarray.h"
#include "partition.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

int
main(void)
{
    acCommInit();
    // Init
    const size_t nn[]       = {4, 4, 4};
    const size_t rr[]       = {1, 1, 1};
    const size_t ndims      = ARRAY_SIZE(nn);
    acHaloExchangeTask task = acHaloExchangeTaskCreate(ndims, nn, rr, 1);
    acHaloExchangeTaskDestroy(&task);

    // Compute
    // acCommHaloExchange(ndims);
    test_math_utils();
    ndarray_test();
    partition_test();

    // Quit
    acCommQuit();

    return EXIT_SUCCESS;
}
