#include <stdio.h>
#include <stdlib.h>

#include "array.h"
#include "comm.h"
#include "math_utils.h"
#include "ndarray.h"
#include "print.h"

static void
get_mm(const size_t ndims, const size_t* nn, const size_t* rr, size_t* mm)
{
    for (size_t i = 0; i < ndims; ++i)
        mm[i] = 2 * rr[i] + nn[i];
}

int
main(void)
{
    acCommInit();

    const size_t nn[]  = {3, 3, 3};
    const size_t rr[]  = {1, 1, 1};
    const size_t ndims = ARRAY_SIZE(nn);

    size_t mm[ndims];
    get_mm(ndims, nn, rr, mm);
    const size_t count = prod(ndims, mm);

    size_t buf0[count];
    size_t buf1[count];

    for (size_t i = 0; i < count; ++i) {
        buf0[i] = i;
        buf1[i] = 2 * i;
    }
    size_t* buffers[]     = {buf0, buf1};
    const size_t nbuffers = ARRAY_SIZE(buffers);
    print("nbuffers", nbuffers);

    HaloExchangeTask* task = acHaloExchangeTaskCreate(ndims, mm, nn, rr, nbuffers);
    acHaloExchangeTaskLaunch(task, nbuffers, buffers);
    acHaloExchangeTaskSynchronize(task);
    acHaloExchangeTaskDestroy(&task);

    acCommQuit();
    return EXIT_SUCCESS;
}
