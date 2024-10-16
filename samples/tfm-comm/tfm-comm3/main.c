#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "misc.h"
#include "print.h"

// #include <mpi.h>
// #include "mpi_utils.h"

#include "math_utils.h"
#include "type_conversion.h"

#include "errchk.h"

int
main(void)
{
    ERRCHK(acCommInit() == ERRORCODE_SUCCESS);

    const uint64_t global_nn[] = {4, 4, 4};
    const uint64_t rr[]        = {1, 1, 1};
    const size_t ndims         = ARRAY_SIZE(global_nn);

    // Setup the communicator
    uint64_t local_nn[ndims], global_nn_offset[ndims];
    ERRCHK(acCommSetup(ndims, global_nn, local_nn, global_nn_offset) == ERRORCODE_SUCCESS);

    ERRCHK(acCommQuit() == ERRORCODE_SUCCESS);
    return EXIT_SUCCESS;
}
