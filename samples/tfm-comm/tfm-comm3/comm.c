#include "comm.h"

#include <mpi.h>

#include <stdio.h>

// #include "errchk_mpi.h"
#include "alloc.h"
#include "errchk.h"
#include "math_utils.h"
#include "type_conversion.h"

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
         ? ERRORCODE_SUCCESS                                                                       \
         : ((handle_mpi_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)),             \
            ERRORCODE_MPI_FAILURE))

ErrorCode
acCommInit(void)
{
    return ERRCHK_MPI_API(MPI_Init(NULL, NULL));
}

#define COMM_MAX_NDIMS ((size_t)4)

typedef struct {
    int* mpi_decomp;
    int* mpi_periods;
    MPI_Comm comm;
} CommCtx;

/** Setup the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
ErrorCode
acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
            uint64_t* global_nn_offset)
{
    ErrorCode errcode = ERRORCODE_SUCCESS;

    // Set MPI errors as non-fatal
    errcode = ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (errcode != ERRORCODE_SUCCESS) goto exit0;

    // Setup nprocs
    int nprocs;
    errcode = ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    if (errcode != ERRORCODE_SUCCESS) goto exit0;

    // Decompose
    int* mpi_decomp = ac_calloc(ndims, sizeof(mpi_decomp[0]));
    errcode         = ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_decomp));
    if (errcode != ERRORCODE_SUCCESS) goto exit1;

    // Create the communicator
    MPI_Comm mpi_comm;
    int* mpi_periods = ac_calloc(ndims, sizeof(mpi_periods[0]));
    set_array_int(1, ndims, mpi_periods); // Fully periodic
    errcode = ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_decomp, mpi_periods, 0, &mpi_comm));
    if (errcode != ERRORCODE_SUCCESS) goto exit2;

exit2:
    ac_free(mpi_periods);
exit1:
    ac_free(mpi_decomp);
exit0:
    return errcode;
}

// ErrorCode
// acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
//             uint64_t* global_nn_offset)
// {
//     int nprocs;
//     int* mpi_decomp  = ac_calloc(ndims, sizeof(mpi_decomp[0]));
//     int* mpi_periods = ac_calloc(ndims, sizeof(mpi_periods[0]));
//     set_array_int(1, ndims, mpi_periods); // Fully periodic

//     ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
//     ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
//     ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_decomp));

//     MPI_Comm mpi_comm;
//     ERRCHK_MPI_API(
//         MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_decomp, mpi_periods, 0, &mpi_comm));

//     ac_free(mpi_periods);
//     ac_free(mpi_decomp);
//     return errcode;
// }

ErrorCode
acCommQuit(void)
{
    return ERRCHK_MPI_API(MPI_Finalize());
}

ErrorCode
acCommGetProcInfo(int* rank, int* nprocs)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}

ErrorCode
acCommBarrier(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}

ErrorCode
acCommPrint(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}

ErrorCode
acCommTest(void)
{
    return ERRORCODE_TEST_FAILURE;
}
