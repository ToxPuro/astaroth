#include "comm.h"

#include <mpi.h>

#include "errchk_mpi.h"
#include "math_utils.h"
#include "misc.h"
#include "nalloc.h"
#include "print.h"
#include "type_conversion.h"

static MPI_Comm comm_ = -1;
// static int _ndims

// static void
// decompose(const size_t ndims, const size_t global_nn, const size_t nlayers,
//           const size_t* partitions_per_layer)
// {
// }

void
acCommInit(const size_t ndims, const size_t* global_nn, size_t* local_nn, size_t* global_nn_offset)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

    // Get nprocs
    int nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // Decompose
    int* mpi_dims;
    ncalloc(ndims, mpi_dims);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    int* mpi_periods;
    nalloc(ndims, mpi_periods);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(MPI_Cart_create(nprocs, as_int(ndims), mpi_dims, mpi_periods, 0, &comm_));

    ndealloc(mpi_periods);
    ndealloc(mpi_dims);
}

void
acCommQuit(void)
{
    ERRCHK_MPI_API(MPI_Comm_free(&comm_));
    ERRCHK_MPI_API(MPI_Finalize());
}

void
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(comm_, rank));
    ERRCHK_MPI_API(MPI_Comm_size(comm_, nprocs));
}

void
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(comm_));
}

void
test_comm(void)
{
    const size_t nn[]  = {2, 2, 2};
    const size_t ndims = ARRAY_SIZE(nn);

    size_t *local_nn, *global_nn_offset;
    nalloc(ndims, local_nn);
    nalloc(ndims, global_nn_offset);

    acCommInit(ndims, nn, local_nn, global_nn_offset);

    int rank, nprocs;
    acCommGetProcInfo(&rank, &nprocs);
    print("rank", rank);
    print("nprocs", nprocs);

    acCommQuit();

    ndealloc(local_nn);
    ndealloc(global_nn_offset);
}
