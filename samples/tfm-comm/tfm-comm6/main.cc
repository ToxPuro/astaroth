#include "comm.h"

#include <cstdlib>

#include "errchk.h"
#include "shape.h"

int
main()
{
    ERRCHK(acCommInit() == ERRORCODE_SUCCESS);

    Shape global_nn;
    Shape local_nn;
    Index local_nn_offset;

    ERRCHK(acCommSetup(3, global_nn.data, local_nn.data, local_nn_offset.data) ==
           ERRORCODE_SUCCESS);
    ERRCHK(acCommBarrier() == ERRORCODE_SUCCESS);
    ERRCHK(acCommPrint() == ERRORCODE_SUCCESS);
    ERRCHK(acCommQuit() == ERRORCODE_SUCCESS);
    return EXIT_SUCCESS;
}
