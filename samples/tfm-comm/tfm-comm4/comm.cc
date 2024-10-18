#include "comm.h"

#include <mpi.h>

#include "errchk.h"
#include "errchk_mpi.h"

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
