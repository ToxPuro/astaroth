#include "comm.h"

#include <mpi.h>

#include <stdio.h>

#include "alloc.h"
#include "errchk.h"
#include "math_utils.h"
#include "type_conversion.h"

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

ErrorCode
acCommInit(void)
{
    return ERRCHK_MPI_API(MPI_Init(NULL, NULL));
}

#include "partition.h"

ErrorCode
acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
            uint64_t* global_nn_offset)
{
    partition_hierarchical();
    return ERRORCODE_NOT_IMPLEMENTED;
}

// #define COMM_MAX_NDIMS ((size_t)4)

// typedef struct {
//     int* mpi_decomp;
//     int* mpi_periods;
//     MPI_Comm comm;
// } CommCtx;

/** Setup the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
// ErrorCode
// acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
//             uint64_t* global_nn_offset)
// {
//     // Set MPI errors as non-fatal
//     ERRCHK_MPI_API(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN))

//     // Setup nprocs
//     int nprocs;
//     ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

//     // Decompose
//     int* mpi_decomp = ac_calloc(ndims, sizeof(mpi_decomp[0]));
//     ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_decomp));

//     // Create the communicator
//     MPI_Comm mpi_comm;
//     int* mpi_periods = ac_calloc(ndims, sizeof(mpi_periods[0]));
//     set_array_int(1, ndims, mpi_periods); // Fully periodic
//     ERRCHK_MPI_API(
//         MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_decomp, mpi_periods, 0, &mpi_comm);

// cleanup_2:
//     ac_free(mpi_periods);
// cleanup_1:
//     ac_free(mpi_decomp);
// done:
//     return errcode;
// }

#include <stdlib.h>

// ErrorCode
// acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
//             uint64_t* global_nn_offset)
// {
//     MPI_Comm parent = MPI_COMM_WORLD;

//     // Set MPI errors as non-fatal
//     ERRCHK_MPI_API(MPI_Comm_set_errhandler(parent, MPI_ERRORS_RETURN));

//     // Setup nprocs
//     int nprocs;
//     ERRCHK_MPI_API(MPI_Comm_size(parent, &nprocs));

//     // Decompose
//     int* mpi_decomp = ac_calloc(ndims, sizeof(mpi_decomp[0]));
//     ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_decomp));

//     // Create the communicator
//     // int* mpi_periods = ac_calloc(ndims, sizeof(mpi_periods[0]));
//     int* mpi_periods = ERRCHK(calloc(ndims, sizeof(mpi_periods[0])));
//     set_array_int(1, ndims, mpi_periods); // Fully periodic
//     MPI_Comm mpi_comm;
//     ERRCHK_MPI_API(MPI_Cart_create(parent, as_int(ndims), mpi_decomp, mpi_periods, 0,
//     &mpi_comm));

//     // int mpi_periods[MAX_NDIMS];
//     // set_array_int(1, ndims, mpi_periods);

//     ac_free(mpi_periods);
//     ac_free(mpi_decomp);
//     return ERRORCODE_SUCCESS;
// }

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
    ERRCHK_MPI_API(MPI_Finalize());
    return ERRORCODE_SUCCESS;
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
