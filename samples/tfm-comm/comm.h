#pragma once
#include <stddef.h>

/** Initialize the communicator module
 * global_nn: dimensions of the global computational domain partitioned to multiple processors
 * local_nn: dimensions of the local computational domain
 * global_nn_offset: offset of the local domain in global scale, e.g.,
 *  global nn index = local nn index + global_nn_offset
 *                  = local nn index + local_nn * decomposition
 * rr: extent of the halo surrounding the computational domain
 */
void acCommInit(const size_t ndims, const size_t* global_nn, size_t* local_nn,
                size_t* global_nn_offset);

void acCommQuit(void);

void acCommGetProcInfo(int* rank, int* nprocs);

void acCommBarrier(void);

// typedef struct HaloExchangeTask_s HaloExchangeTask;

// HaloExchangeTask* acHaloExchangeTaskCreate(const size_t ndims, const size_t* mm, const size_t*
// nn,
//                                            const size_t* rr, const size_t nbuffers);

// void acHaloExchangeTaskLaunch(const HaloExchangeTask* task, const size_t nbuffers,
//                               size_t* buffers[nbuffers]);

// void acHaloExchangeTaskSynchronize(const HaloExchangeTask* task);

// void acHaloExchangeTaskDestroy(HaloExchangeTask** task);

void test_comm(void);

/*
// Require that local_nn is also surrounded by rr.
// This is implied:
acCommInit(global_nn, rr, &local_nn)
AcDeviceCreate(nn);
DeviceGlobalOffset

*/
