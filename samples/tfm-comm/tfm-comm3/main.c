#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "misc.h"

int
main(void)
{
    acCommInit();

    const uint64_t global_nn[] = {4, 6};
    const uint64_t rr[]        = {1, 2, 1, 1, 1};
    const size_t ndims         = ARRAY_SIZE(global_nn);

    uint64_t local_nn[ndims], global_nn_offset[ndims];
    acCommSetup(ndims, global_nn, local_nn, global_nn_offset);

    acCommQuit();
    return EXIT_SUCCESS;
}
