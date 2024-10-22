#include "comm.h"

#include <cstdlib>

#include "errchk.h"
#include "shape.h"

int
main()
{
    acCommInit();

    Shape global_nn;
    Shape local_nn;
    Index local_nn_offset;

    acCommSetup(3, global_nn.data, local_nn.data, local_nn_offset.data);
    acCommBarrier();
    acCommPrint();

    // Setup the halo exchange
    acCommTest();

    acCommQuit();
    return EXIT_SUCCESS;
}
