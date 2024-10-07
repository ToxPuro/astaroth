#pragma once
#include <stddef.h>

/** Initialize the communicator module.
 * Should be called as early in the program as possible to avoid interference
 * with the MPI implementation, such as opening files. Recommended by, e.g., MPICH.
 */
void acCommInit(void);

/** Setup the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
void acCommSetup(const size_t ndims, const size_t* global_nn, size_t* local_nn,
                 size_t* global_nn_offset);

void acCommQuit(void);

void acCommGetProcInfo(int* rank, int* nprocs);

void acCommBarrier(void);

void print_comm(void);

typedef struct HaloSegmentBatch_s* HaloSegmentBatch;

HaloSegmentBatch halo_segment_batch_create(const size_t ndims, const size_t* local_mm,
                                           const size_t* local_nn, const size_t* local_nn_offset,
                                           const size_t n_aggregate_buffers);

void halo_segment_batch_destroy(HaloSegmentBatch* batch);

void halo_segment_batch_launch(HaloSegmentBatch batch);

void halo_segment_batch_wait(HaloSegmentBatch batch);

void test_comm(void);
