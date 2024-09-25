#pragma once
#include <stddef.h>

void acCommInit(void);

void acCommQuit(void);

typedef struct HaloExchangeTask_s HaloExchangeTask;

HaloExchangeTask* acHaloExchangeTaskCreate(const size_t ndims, const size_t* mm, const size_t* nn,
                                           const size_t* rr, const size_t nbuffers);

void acHaloExchangeTaskLaunch(const HaloExchangeTask* task, const size_t nbuffers,
                              size_t* buffers[nbuffers]);

void acHaloExchangeTaskSynchronize(const HaloExchangeTask* task);

void acHaloExchangeTaskDestroy(HaloExchangeTask** task);
