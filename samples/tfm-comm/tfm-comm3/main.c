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
    acCommInit();

    const uint64_t global_nn[] = {4, 4, 4};
    const uint64_t rr[]        = {1, 1, 1};
    const size_t ndims         = ARRAY_SIZE(global_nn);

    // Setup the communicator
    uint64_t local_nn[ndims], global_nn_offset[ndims];
    acCommSetup(ndims, global_nn, local_nn, global_nn_offset);

    // printd(ndims);
    // printd_array(ndims, global_nn);
    // printd_array(ndims, rr);
    // printd_array(ndims, local_nn);
    // printd_array(ndims, global_nn_offset);

    // // Setup the local mesh
    // const size_t count = as_size_t(prod(ndims, global_nn));
    // printd(count);

    // Dims global_nn(4, 4, 4);
    // Dims rr(1, 1, 1);
    // Domaininfo info = acCommSetup()

    // the local buffer is not decoupled from the comm implementaiton
    //

    // const uint64_t global_nn[] = {4, 6};
    // const uint64_t rr[]        = {1, 1, 1, 1, 1};
    // const size_t ndims         = ARRAY_SIZE(global_nn);

    // uint64_t local_nn[ndims], global_nn_offset[ndims];
    // acCommSetup(ndims, global_nn, local_nn, global_nn_offset);

    acCommQuit();
    return EXIT_SUCCESS;
}
