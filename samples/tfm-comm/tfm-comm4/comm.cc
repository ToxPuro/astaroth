#include "comm.h"

#include <csignal>
#include <unistd.h>

#include <mpi.h>

#include "errchk_mpi.h"
#include "mpi_utils.h"
#include "print.h"
#include "type_conversion.h"

#include "static_array.h"

/*
 * Global variables
 */

namespace Comm {
MPI_Comm comm       = MPI_COMM_NULL;
MPI_Datatype dtype  = MPI_DATATYPE_NULL;
int ndims           = 0;
bool setup_complete = false;
} // namespace Comm

/*
 * Signal handling
 */
static void
signal_handler(int signum)
{
    const char msg[] = "SIGABRT received\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
    MPI_Abort(MPI_COMM_WORLD, signum);
}

/*
 * Communication module
 */
ErrorCode
acCommInit(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    signal(SIGABRT, signal_handler); // Setup signal handler for errors
    return ERRORCODE_NOT_IMPLEMENTED;
}

// typedef struct StaticArray<int, NTUPLE_MAX_NDIMS> Ntuple_mpi;
// typedef struct StaticArray<uint64_t, NTUPLE_MAX_NDIMS> Ntuple;

/** Setup the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
ErrorCode
acCommSetup(const size_t ndims, const uint64_t* global_nn_ptr, uint64_t* local_nn_ptr,
            uint64_t* global_nn_offset_ptr)
{
    if (Comm::setup_complete) {
        WARNING("acCommSetup was called more than once. This is not allowed.");
        return ERRORCODE_INPUT_FAILURE;
    }
    ERRCHK(ndims <= NTUPLE_MAX_NDIMS);
    ERRCHK(global_nn_ptr);
    ERRCHK(local_nn_ptr);
    ERRCHK(global_nn_offset_ptr);

    // Set MPI errors of parent temporarily as non-fatal
    const MPI_Comm parent = MPI_COMM_WORLD;
    MPI_Errhandler parent_errhandler;
    ERRCHK_MPI_API(MPI_Comm_get_errhandler(parent, &parent_errhandler));
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(parent, MPI_ERRORS_RETURN));

    // Get nprocs
    int mpi_nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(parent, &mpi_nprocs));

    // Decompose
    Ntuple<int> mpi_decomp(ndims);
    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(mpi_decomp.count), mpi_decomp.data));

    // Create the communicator
    Ntuple<int> mpi_periods(ndims, 1); // Set to fully periodic
    ERRCHK_MPI_API(
        MPI_Cart_create(parent, as<int>(ndims), mpi_decomp.data, mpi_periods.data, 0, &Comm::comm));

    // Set MPI errors of the local communicator as non-fatal
    // and set error handling of the parent communicator its original setting
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(Comm::comm, MPI_ERRORS_RETURN));
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(parent, parent_errhandler));

    // Get coordinates
    int mpi_rank;
    ERRCHK_MPI_API(MPI_Comm_rank(Comm::comm, &mpi_rank));

    Ntuple<int> mpi_coords(ndims);
    ERRCHK_MPI_API(MPI_Cart_coords(Comm::comm, mpi_rank, as<int>(ndims), mpi_coords.data));
    const Ntuple<uint64_t> coords(mpi_coords.reversed());

    // Compute the local problem size
    Ntuple<uint64_t> global_nn(ndims, global_nn_ptr);
    Ntuple<uint64_t> decomp(mpi_decomp.reversed());
    const auto local_nn         = global_nn / decomp;
    const auto global_nn_offset = coords * local_nn;
    std::copy(local_nn.data, local_nn.data + ndims, local_nn_ptr);
    std::copy(global_nn_offset.data, global_nn_offset.data + ndims, global_nn_offset_ptr);

    // Set other
    Comm::dtype = MPI_DOUBLE;
    Comm::ndims = as<int>(ndims);

    Comm::setup_complete = true;
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommQuit(void)
{
    if (Comm::comm != MPI_COMM_NULL)
        ERRCHK_MPI_API(MPI_Comm_free(&Comm::comm));
    Comm::comm  = MPI_COMM_NULL;
    Comm::dtype = MPI_DATATYPE_NULL;
    Comm::ndims = 0;

    ERRCHK_MPI_API(MPI_Finalize());
    signal(SIGABRT, SIG_DFL); // Reset signal handler to default
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(Comm::comm, rank));
    ERRCHK_MPI_API(MPI_Comm_size(Comm::comm, nprocs));
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(Comm::comm));
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommPrint(void)
{
    int rank, nprocs;
    ERRCHK_MPI(acCommGetProcInfo(&rank, &nprocs) == ERRORCODE_SUCCESS);

    Ntuple<int> mpi_decomp(Comm::ndims), mpi_periods(Comm::ndims), mpi_coords(Comm::ndims);
    ERRCHK_MPI_API(
        MPI_Cart_get(Comm::comm, Comm::ndims, mpi_decomp.data, mpi_periods.data, mpi_coords.data));

    MPI_SYNCHRONOUS_BLOCK_START(Comm::comm)
    PRINTD(mpi_decomp);
    PRINTD(mpi_periods);
    PRINTD(mpi_coords);
    MPI_SYNCHRONOUS_BLOCK_END(Comm::comm)

    return ERRORCODE_SUCCESS;
}

/**
 * Test the comm functions.
 * Returns 0 on success and the number of errors encountered otherwise.
 */
ErrorCode
acCommTest(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}
