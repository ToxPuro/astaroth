#include "comm.h"

#include <limits.h> // INT_MAX
#include <mpi.h>

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include "errchk.h"
#include "halo_segment_batch.h"
#include "math_utils.h"
#include "print.h"

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
acHaloExchangeTaskDestroy(HaloExchangeTask** task)
{
    acHaloSegmentBatchDestroy(&((*task)->batch));
    free(*task);
    *task = NULL;
}

/**
 * At each call, returns the next integer in range [0, INT_MAX].
 * If the counter overflows, the counter wraps around and starts again from 0.
 * NOTE: Not thread-safe.
 */
static int
get_tag(void)
{
    static int counter = -1;
    counter            = next_positive_integer(counter);
    return counter;
}

static void
test_get_tag(void)
{
    {
        int prev = get_tag();
        for (size_t i = 0; i < 1000; ++i) {
            int curr = get_tag();
            if (prev == INT_MAX)
                ERRCHK(curr == 0)
            else
                ERRCHK(curr > prev);
            prev = curr;
        }
    }
}

void
acHaloExchangeTaskLaunch(const HaloExchangeTask* task, const size_t nbuffers,
                         size_t* buffers[nbuffers])
{
    test_get_tag();
}

void
acHaloExchangeTaskSynchronize(const HaloExchangeTask* task)
{
}

void
test_comm(void)
{
    test_get_tag();
}
