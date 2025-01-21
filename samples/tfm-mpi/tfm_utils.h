#pragma once

#include "acc_runtime.h"

typedef struct {
    char* config_path;
} Arguments;

#ifdef __cplusplus
extern "C" {
#endif

int acParseArguments(const int argc, char* argv[], Arguments* args);

int acParseINI(const char* filepath, AcMeshInfo* info);

int acPrintArguments(const Arguments args);

int acHostUpdateLocalBuiltinParams(AcMeshInfo* config);

int acHostUpdateForcingParams(AcMeshInfo* info);

int acHostUpdateMHDSpecificParams(AcMeshInfo* info);

int acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info);

int acPrintMeshInfoTFM(const AcMeshInfo config);

AcReal calc_timestep(const AcReal uumax, const AcReal vAmax, const AcReal shock_max,
                     const AcMeshInfo info);

#ifdef __cplusplus
}
#endif
