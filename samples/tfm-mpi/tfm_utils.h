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

void acPrintArguments(const Arguments args);

int acHostUpdateBuiltinParams(AcMeshInfo* config);

int acHostUpdateMHDSpecificParams(AcMeshInfo* info);

int acHostUpdateTFMSpecificGlobalParams(AcMeshInfo* info);

void acPrintMeshInfo(const AcMeshInfo config);

#ifdef __cplusplus
}
#endif
