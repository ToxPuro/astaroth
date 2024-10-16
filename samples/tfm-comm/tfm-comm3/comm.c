#include "comm.h"

#include <mpi.h>

#include <stdio.h>

#include "alloc.h"
#include "errchk.h"
#include "math_utils.h"
#include "type_conversion.h"

#include "mpi_utils.h"
#include "print.h"

/*
 * Global variables
 */
static MPI_Comm mpi_comm_            = MPI_COMM_NULL;
static const MPI_Datatype mpi_dtype_ = MPI_DOUBLE;

/*
 * Error handling
 */
static void
handle_error(const char* function, const char* file, const long line, const char* expression)
{
    errchk_print_error(function, file, line, expression, "");
    MPI_Abort(MPI_COMM_WORLD, -1);
}

static void
handle_mpi_error(const int errorcode, const char* function, const char* file, const long line,
                 const char* expression)
{
    char description[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errorcode, description, &resultlen);
    errchk_print_error(function, file, line, expression, description);
    MPI_Abort(MPI_COMM_WORLD, -1);
}

#define ERRCHK_MPI_API(errorcode)                                                                  \
    (((errorcode) == MPI_SUCCESS)                                                                  \
         ? 0                                                                                       \
         : ((handle_mpi_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)), -1))

#define ERRCHK_MPI(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))

/*
 * Communicator implementation
 */
ErrorCode
acCommInit(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
            uint64_t* global_nn_offset)
{
    const MPI_Comm parent = MPI_COMM_WORLD;

    const int mpi_ndims = as_int(ndims);

    // Setup nprocs
    int mpi_nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(parent, &mpi_nprocs));

    // Decompose
    int* mpi_decomp = ac_calloc(ndims, sizeof(mpi_decomp[0]));
    ERRCHK_MPI(mpi_decomp);
    ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, mpi_ndims, mpi_decomp));

    // Create the communicator
    int* mpi_periods = ac_calloc(ndims, sizeof(mpi_periods[0]));
    ERRCHK_MPI(mpi_periods);
    set_array_int(1, ndims, mpi_periods); // Fully periodic
    ERRCHK_MPI_API(MPI_Cart_create(parent, mpi_ndims, mpi_decomp, mpi_periods, 0, &mpi_comm_));

    // Set MPI errors as non-fatal
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(mpi_comm_, MPI_ERRORS_RETURN));

    // Compute the local problem size
    uint64_t* decomp = ac_calloc(ndims, sizeof(decomp[0]));
    ERRCHK_MPI(decomp);
    as_astaroth_format(ndims, mpi_decomp, decomp);
    array_div(ndims, global_nn, decomp, local_nn);

    // Compute the global offset
    int mpi_rank;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &mpi_rank));

    int* mpi_coords = ac_calloc(ndims, sizeof(mpi_coords[0]));
    ERRCHK_MPI(mpi_coords);
    ERRCHK_MPI_API(MPI_Cart_coords(mpi_comm_, mpi_rank, mpi_ndims, mpi_coords));

    uint64_t* coords = ac_calloc(ndims, sizeof(coords[0]));
    ERRCHK_MPI(mpi_coords);
    as_astaroth_format(ndims, mpi_coords, coords);
    array_mul(ndims, coords, local_nn, global_nn_offset);

    // Cleanup
    ac_free((void**)&coords);
    ac_free((void**)&mpi_coords);
    ac_free((void**)&decomp);
    ac_free((void**)&mpi_periods);
    ac_free((void**)&mpi_decomp);
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommQuit(void)
{
    if (mpi_comm_ != MPI_COMM_NULL)
        ERRCHK_MPI_API(MPI_Comm_free(&mpi_comm_));

    ERRCHK_MPI_API(MPI_Finalize());
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommGetProcInfo(int* mpi_rank, int* mpi_nprocs)
{
    *mpi_rank   = 0;
    *mpi_nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, mpi_rank));
    ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, mpi_nprocs));
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(mpi_comm_));
    return ERRORCODE_SUCCESS;
}

ErrorCode
acCommTest(void)
{
    return ERRORCODE_TEST_FAILURE;
}
