#include "comm.h"

#include <cstdlib>

#include "errchk.h"
#include "ndarray.h"
#include "shape.h"

int
main()
{
    acCommInit();

    Shape global_nn = {4, 4, 4};
    Shape local_nn(global_nn.count);
    Index local_nn_offset(global_nn.count);

    acCommSetup(global_nn.count, global_nn.data, local_nn.data, local_nn_offset.data);
    acCommBarrier();
    acCommPrint();

    // Setup the halo exchange
    acCommTest();

    acCommQuit();
    return EXIT_SUCCESS;
}
