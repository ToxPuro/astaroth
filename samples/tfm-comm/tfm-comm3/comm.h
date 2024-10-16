#pragma once
#include <stddef.h>
#include <stdint.h>

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the communicator module.
 * Should be called as early in the program as possible to avoid interference
 * with the MPI implementation, such as opening files. Recommended by, e.g., MPICH.
 */
ErrorCode acCommInit(void);

/** Setup the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
ErrorCode acCommSetup(const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
                      uint64_t* global_nn_offset);

ErrorCode acCommQuit(void);

ErrorCode acCommGetProcInfo(int* rank, int* nprocs);

ErrorCode acCommBarrier(void);

/**
 * Test the comm functions.
 * Returns 0 on success and the number of errors encountered otherwise.
 */
ErrorCode acCommTest(void);

#ifdef __cplusplus
}
#endif
