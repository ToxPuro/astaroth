#include "comm.h"

#include <cstdlib>

#include "errchk.h"
#include "shape.h"

int
main()
{
    acCommInit();

    const size_t ndims = 3;
    Shape global_nn(ndims);
    Shape local_nn(ndims);
    Index local_nn_offset(ndims);

    acCommSetup(3, global_nn.data, local_nn.data, local_nn_offset.data);
    acCommBarrier();
    acCommPrint();

    // Setup the halo exchange
    acCommTest();

    acCommQuit();
    return EXIT_SUCCESS;
}
