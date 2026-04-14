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

AC_BEGIN_C_DECLARATIONS

/** Generates PC-inspired forcing params (vaisala_interactionoflarge_2021:
 * https://doi.org/10.3847/1538-4357/abceca) */
// ForcingParams generateForcingParams(const AcReal relhel, const AcReal magnitude, const AcReal
// kmin,
//                                     const AcReal kmax);

/** Generates exact PC forcing params based on PC user manual,
 * brandenburg_crosshelically_2019 10.1002/asna.201913602, and used in pekkila_gpuaccelerated_2025
 */
ForcingParams generateHelicalForcingParams(const AcReal relhel, const AcReal magnitude,
                                           const AcReal kmin, const AcReal kmax);

int loadForcingParamsToMeshInfo(const ForcingParams forcing_params, AcMeshInfo* info);

void printForcingParams(const ForcingParams forcing_params);

AC_END_C_DECLARATIONS
