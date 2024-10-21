#include "comm.h"

#include "segment.h"

#include "print.h"

#include "vecn.h"

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

    VecN<int, 3> a = {-1, 2, 3};
    VecN<uint64_t, 3> b(a);
    PRINTD(a);
    PRINTD(b);

    acCommSetup(global_nn.count, global_nn.data, local_nn.data, global_nn_offset.data);
    acCommPrint();
    acCommQuit();
    return 0;
}
