#include <iostream>
#include <stdexcept>

#include "comm.h"

#include "print.h"
#include "shape.h"

int
main(void)
{
    acCommInit();

    Shape global_nn = {4, 4};
    Shape local_nn(global_nn.count);
    Index global_nn_offset(global_nn.count);
    PRINTD(global_nn);
    PRINTD(local_nn);
    PRINTD(global_nn_offset);

    acCommSetup(global_nn.count, global_nn.data, local_nn.data, global_nn_offset.data);
    acCommPrint();
    acCommQuit();

    return EXIT_SUCCESS;
}
