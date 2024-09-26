#include "comm.h"

#include <mpi.h>

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include "errchk_mpi.h"
#include "halo_segment_batch.h"
#include "math_utils.h"
#include "print.h"

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
acHaloExchangeTaskDestroy(HaloExchangeTask** task)
{
    acHaloSegmentBatchDestroy(&((*task)->batch));
    free(*task);
    *task = NULL;
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
