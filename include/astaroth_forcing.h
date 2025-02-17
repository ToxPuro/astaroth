#pragma once

#include "acc_runtime.h"

typedef struct {
    AcReal magnitude;
    AcReal3 k_force;
    AcReal3 ff_hel_re;
    AcReal3 ff_hel_im;
    AcReal phase;
    AcReal kaver;
} ForcingParams;

#ifdef __cplusplus
extern "C" {
#endif

ForcingParams generateForcingParams(const AcReal relhel, const AcReal magnitude, const AcReal kmin,
                                    const AcReal kmax);

int loadForcingParamsToMeshInfo(const ForcingParams forcing_params, AcMeshInfo* info);

void printForcingParams(const ForcingParams forcing_params);

#ifdef __cplusplus
} // extern "C"
#endif
