#pragma once
#include <stddef.h>

// typedef struct acHaloExchangeTask_s acHaloExchangeTask;

#include "halo_segment_batch.h"

void acCommInit(void);
void acCommQuit(void);

typedef struct {
    HaloSegmentBatch batch;
} acHaloExchangeTask;

acHaloExchangeTask acHaloExchangeTaskCreate(const size_t ndims, const size_t* local_mm,
                                            const size_t* rr, const size_t nbuffers);

void acHaloExchangeTaskLaunch(const acHaloExchangeTask task, const size_t nbuffers,
                              const size_t* buffers);

void acHaloExchangeTaskSynchronize(const acHaloExchangeTask task);

void acHaloExchangeTaskDestroy(acHaloExchangeTask* task);
