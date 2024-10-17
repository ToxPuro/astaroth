#include "comm.h"

#include <mpi.h>

#include "errchk.h"

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
#define ERRCHK_MPI(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))

/*
 * Communication module
 */
ErrorCode
acCommInit(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}

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
    return ERRORCODE_NOT_IMPLEMENTED;
}

ErrorCode
acCommQuit(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
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

/**
 * Test the comm functions.
 * Returns 0 on success and the number of errors encountered otherwise.
 */
ErrorCode
acCommTest(void)
{
    return ERRORCODE_NOT_IMPLEMENTED;
}
