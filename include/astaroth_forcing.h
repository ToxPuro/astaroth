#pragma once

#include "datatypes.h"

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

ForcingParams generateForcingParams(const AcReal relhel, const AcReal kmin, const AcReal kmax);

#ifdef __cplusplus
} // extern "C"
#endif