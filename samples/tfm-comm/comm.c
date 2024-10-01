#include "comm.h"

#include <mpi.h>

#define SUCCESS (0)
#define FAILURE (-1)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include "array.h"
#include "errchk_mpi.h"
#include "halo_segment_batch.h"
#include "math_utils.h"
#include "misc.h"
#include "mpi_utils.h"
#include "print.h"
#include "type_conversion.h"

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

void
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, rank));
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, nprocs));
}

void
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
}

struct HaloExchangeTask_s {
    HaloSegmentBatch batch;

    size_t ndims;
    size_t* mm;
    size_t* nn;
    size_t* rr;

    MPI_Comm comm_cart;
};

static void
dims_create(const int nprocs, const size_t ndims, size_t* dims, size_t* periods)
{
    int mpi_dims[ndims], mpi_periods[ndims];
    iset(0, ndims, mpi_dims);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    to_astaroth_format(ndims, mpi_dims, dims);
    to_astaroth_format(ndims, mpi_periods, periods);
    ERRCHK_MPI(prod(ndims, dims) == as_size_t(nprocs));
}

HaloExchangeTask*
acHaloExchangeTaskCreate(const size_t ndims, const size_t* mm, const size_t* nn, const size_t* rr,
                         const size_t nbuffers)
{
    HaloExchangeTask* task = malloc(sizeof(HaloExchangeTask));
    ERRCHK(task != NULL);

    task->batch = acHaloSegmentBatchCreate(ndims, mm, nn, rr, nbuffers);

    task->ndims = ndims;
    task->mm    = malloc(sizeof(task->mm[0]) * ndims);
    task->nn    = malloc(sizeof(task->nn[0]) * ndims);
    task->rr    = malloc(sizeof(task->rr[0]) * ndims);
    copy(ndims, mm, task->mm);
    copy(ndims, nn, task->nn);
    copy(ndims, rr, task->rr);

    // Get nprocs
    int nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // Decompose
    size_t dims[ndims], periods[ndims];
    dims_create(nprocs, ndims, dims, periods);

    // Create communicator
    int mpi_dims[ndims], mpi_periods[ndims];
    to_mpi_format(ndims, dims, mpi_dims);
    to_mpi_format(ndims, periods, mpi_periods);
    ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &task->comm_cart));

    return task;
}

void
acHaloExchangeTaskDestroy(HaloExchangeTask** task)
{
    ERRCHK_MPI_API(MPI_Comm_free(&(*task)->comm_cart));
    free((*task)->rr);
    free((*task)->nn);
    free((*task)->mm);
    acHaloSegmentBatchDestroy(&((*task)->batch));
    free(*task);
    *task = NULL;
}

static void
get_mpi_coords_forward(const size_t ndims, const size_t* nn, const size_t* nn_offset,
                       const size_t* offset, const int* mpi_coords, int* mpi_neighbor_coords)
{
    int mpi_nn[ndims], mpi_nn_offset[ndims], mpi_offset[ndims];
    to_mpi_format(ndims, nn, mpi_nn);
    to_mpi_format(ndims, nn_offset, mpi_nn_offset);
    to_mpi_format(ndims, offset, mpi_offset);

    copyi(ndims, mpi_coords, mpi_neighbor_coords);

    for (size_t i = 0; i < ndims; ++i) {
        if (mpi_offset[i] < mpi_nn_offset[i])
            mpi_neighbor_coords[i] -= 1;
        else if (mpi_offset[i] >= mpi_nn_offset[i] + mpi_nn[i])
            mpi_neighbor_coords[i] += 1;
    }
}

static void
get_mpi_coords_backward(const size_t ndims, const size_t* nn, const size_t* nn_offset,
                        const size_t* offset, const int* mpi_coords, int* mpi_neighbor_coords)
{
    int mpi_nn[ndims], mpi_nn_offset[ndims], mpi_offset[ndims];
    to_mpi_format(ndims, nn, mpi_nn);
    to_mpi_format(ndims, nn_offset, mpi_nn_offset);
    to_mpi_format(ndims, offset, mpi_offset);

    copyi(ndims, mpi_coords, mpi_neighbor_coords);

    for (size_t i = 0; i < ndims; ++i) {
        if (mpi_offset[i] < mpi_nn_offset[i])
            mpi_neighbor_coords[i] += 1;
        else if (mpi_offset[i] >= mpi_nn_offset[i] + mpi_nn[i])
            mpi_neighbor_coords[i] -= 1;
    }
}

void
acHaloExchangeTaskLaunch(const HaloExchangeTask* task, const size_t nbuffers,
                         size_t* buffers[nbuffers])
{

    const HaloSegmentBatch* batch = &task->batch;
    const size_t ndims            = task->ndims;
    const size_t* mm              = task->mm;
    const size_t* nn              = task->nn;
    const size_t* rr              = task->rr;
    MPI_Comm comm_cart            = task->comm_cart;

    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(comm_cart, &rank));

    for (size_t i = 0; i < batch->npackets; ++i) {

        // Offset and subarrays
        const size_t* offset             = batch->local_packets[i].offset;
        const MPI_Datatype send_subarray = batch->send_subarrays[i];
        const MPI_Datatype recv_subarray = batch->recv_subarrays[i];

        // Coordinates
        int mpi_coords[ndims];
        ERRCHK_MPI_API(MPI_Cart_coords(comm_cart, rank, as_int(ndims), mpi_coords));

        int mpi_coords_send_neighbor[ndims], send_neighbor;
        get_mpi_coords_forward(ndims, nn, rr, offset, mpi_coords, mpi_coords_send_neighbor);
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_send_neighbor, &send_neighbor));

        int mpi_coords_recv_neighbor[ndims], recv_neighbor;
        get_mpi_coords_backward(ndims, nn, rr, offset, mpi_coords, mpi_coords_recv_neighbor);
        ERRCHK_MPI_API(MPI_Cart_rank(comm_cart, mpi_coords_recv_neighbor, &recv_neighbor));

        for (size_t j = 0; j < nbuffers; ++j) {
            const int tag = get_tag();
            // ERRCHK_MPI_API(MPI_Isendrecv(buffers[j], 1, send_subarray, send_neighbor, tag,
            //                              buffers[j], 1, recv_subarray, recv_neighbor, tag,
            //                              comm_cart, &batch->requests[i]));
            ERRCHK_MPI_API(MPI_Isend(buffers[j], 1, send_subarray, send_neighbor, tag, comm_cart,
                                     &batch->send_reqs[i]));
            ERRCHK_MPI_API(MPI_Irecv(buffers[j], 1, recv_subarray, recv_neighbor, tag, comm_cart,
                                     &batch->recv_reqs[i]));
        }
    }
}

void
acHaloExchangeTaskSynchronize(const HaloExchangeTask* task)
{
    acHaloSegmentBatchWait(task->batch);
}
