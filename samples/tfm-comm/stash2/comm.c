#include "comm.h"

#include <mpi.h>

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include "errchk.h"
#include "halo_segment_batch.h"

#define ERRCHK_MPI(retval)                                                                         \
    {                                                                                              \
        ERRCHK(retval);                                                                            \
        if ((retval) == 0) {                                                                       \
            MPI_Abort(MPI_COMM_WORLD, 0);                                                          \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define ERRCHK_MPI_API(errorcode)                                                                  \
    if ((errorcode) != MPI_SUCCESS) {                                                              \
        char description[MPI_MAX_ERROR_STRING];                                                    \
        int resultlen;                                                                             \
        MPI_Error_string(errorcode, description, &resultlen);                                      \
        ERRCHKK((errorcode) == MPI_SUCCESS, description);                                          \
        MPI_Abort(MPI_COMM_WORLD, 0);                                                              \
        exit(EXIT_FAILURE);                                                                        \
    }

// Disable all MPI API calls
// #undef ERRCHK_MPI_API
// #define ERRCHK_MPI_API(x)

void
acCommInit(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
}

void
acCommQuit(void)
{
    ERRCHK_MPI_API(MPI_Finalize());
}

static void
get_nn(const size_t ndims, const size_t* mm, const size_t* rr, size_t* nn)
{
    for (size_t i = 0; i < ndims; ++i)
        nn[i] = mm[i] - 2 * rr[i];
}

acHaloExchangeTask
acHaloExchangeTaskCreate(const size_t ndims, const size_t* local_mm, const size_t* rr,
                         const size_t nfields)
{
    size_t local_nn[ndims];
    get_nn(ndims, local_mm, rr, local_nn);
    acHaloExchangeTask task = (acHaloExchangeTask){
        .batch = acHaloSegmentBatchCreate(ndims, local_nn, rr, nfields),
    };

    return task;
}

void
acHaloExchangeTaskLaunch(const acHaloExchangeTask task, const size_t nbuffers,
                         const size_t* buffers)
{
    return;
}

void
acHaloExchangeTaskSynchronize(const acHaloExchangeTask task)
{
    return;
}

void
acHaloExchangeTaskDestroy(acHaloExchangeTask* task)
{
    acHaloSegmentBatchDestroy(&task->batch);
    return;
}
