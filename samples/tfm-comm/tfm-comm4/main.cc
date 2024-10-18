#include "comm.h"

#include "static_array.h"

#include "segment.h"

#include "print.h"

int
main(void)
{
    acCommInit();

    Ntuple<uint64_t> global_nn = {4, 4};
    Ntuple<uint64_t> local_nn(global_nn.count);
    Ntuple<uint64_t> global_nn_offset(global_nn.count);

    Shape dims    = {1, 2, 3};
    Shape subdims = {1, 2, 3};
    Index offset  = {4, 5, 6};
    Segment segment(dims, subdims, offset);
    PRINTD(segment);

    acCommSetup(global_nn.count, global_nn.data, local_nn.data, global_nn_offset.data);
    acCommPrint();
    acCommQuit();
    return 0;
}
