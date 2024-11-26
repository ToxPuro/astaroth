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

int acHostUpdateMHDSpecificParams(AcMeshInfo* info);

int acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info);

int acVerifyMeshInfo(const AcMeshInfo info);

int acPrintMeshInfo(const AcMeshInfo config);

#ifdef __cplusplus
}
#endif
