#include "comm.h"

#include <mpi.h>

#include "errchk_mpi.h"
#include "math_utils.h"
#include "misc.h"
#include "mpi_utils.h"
#include "nalloc.h"
#include "print.h"
#include "type_conversion.h"

static MPI_Comm mpi_comm_ = MPI_COMM_NULL;
static int mpi_ndims_     = -1;

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

    // Create the primary communicator
    int* mpi_periods;
    nalloc(ndims, mpi_periods);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &mpi_comm_));

    // Setup the rest of the global variables
    mpi_ndims_ = as_int(ndims);

    // Compute local_nn
    size_t* dims;
    nalloc(ndims, dims);
    to_astaroth_format(ndims, mpi_dims, dims);
    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];
    ndealloc(dims);

    // Compute global_nn_offset
    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));

    int* mpi_coords;
    ncalloc(ndims, mpi_coords);
    ERRCHK_MPI_API(MPI_Cart_coords(mpi_comm_, rank, mpi_ndims_, mpi_coords));

    size_t* coords;
    nalloc(ndims, coords);
    to_astaroth_format(ndims, mpi_coords, coords);
    for (size_t i = 0; i < ndims; ++i)
        global_nn_offset[i] = local_nn[i] * coords[i];
    ndealloc(coords);

    ndealloc(mpi_coords);

    // Cleanup
    ndealloc(mpi_periods);
    ndealloc(mpi_dims);
}

void
acCommQuit(void)
{
    ERRCHK_MPI_API(MPI_Comm_free(&mpi_comm_));
    ERRCHK_MPI_API(MPI_Finalize());
}

void
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, rank));
    ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, nprocs));
}

void
print_comm(void)
{
    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, &nprocs));

    int *mpi_dims, *mpi_periods, *mpi_coords;
    ncalloc(as_size_t(mpi_ndims_), mpi_dims);
    ncalloc(as_size_t(mpi_ndims_), mpi_periods);
    ncalloc(as_size_t(mpi_ndims_), mpi_coords);
    ERRCHK_MPI_API(MPI_Cart_get(mpi_comm_, mpi_ndims_, mpi_dims, mpi_periods, mpi_coords));
    // ERRCHK_MPI_API(MPI_Cart_coords(mpi_comm_, rank, mpi_ndims_, mpi_coords));

    for (int i = 0; i < nprocs; ++i) {
        acCommBarrier();
        if (rank == i) {
            printf("Rank %d of %d:\n", rank, nprocs);
            print_array("\t mpi_dims", as_size_t(mpi_ndims_), mpi_dims);
            print_array("\t mpi_periods", as_size_t(mpi_ndims_), mpi_periods);
            print_array("\t mpi_coords", as_size_t(mpi_ndims_), mpi_coords);
        }
        acCommBarrier();
    }
    ndealloc(mpi_dims);
    ndealloc(mpi_periods);
    ndealloc(mpi_coords);
}

void
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(mpi_comm_));
}

void
test_comm(void)
{
    const size_t nn[]  = {8, 8, 8};
    const size_t ndims = ARRAY_SIZE(nn);

    size_t *local_nn, *global_nn_offset;
    nalloc(ndims, local_nn);
    nalloc(ndims, global_nn_offset);

    acCommInit(ndims, nn, local_nn, global_nn_offset);

    print_comm();
    // print_array("local_nn", ndims, local_nn);
    // print_array("global_nn_offset", ndims, global_nn_offset);

    acCommQuit();

    ndealloc(local_nn);
    ndealloc(global_nn_offset);
}
