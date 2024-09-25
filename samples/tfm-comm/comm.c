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
#undef ERRCHK_MPI_API
#define ERRCHK_MPI_API(x)

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

struct HaloExchangeTask_s {
    HaloSegmentBatch batch;
};

HaloExchangeTask*
acHaloExchangeTaskCreate(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* rr,
                         const size_t nbuffers)
{
    HaloExchangeTask* task = malloc(sizeof(HaloExchangeTask));
    ERRCHK(task != NULL);

    task->batch = acHaloSegmentBatchCreate(ndims, mm, nn, rr, nbuffers);

    return task;
}

void
acHaloExchangeTaskLaunch(const HaloExchangeTask* task, const size_t nbuffers,
                         size_t* buffers[nbuffers])
{
}

void
acHaloExchangeTaskSynchronize(const HaloExchangeTask* task)
{
}

void
acHaloExchangeTaskDestroy(HaloExchangeTask** task)
{
    acHaloSegmentBatchDestroy(&((*task)->batch));
    free(*task);
    *task = NULL;
}
