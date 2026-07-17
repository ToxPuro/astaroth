#pragma once
#include "device_headers.h"
#include "acreal.h"
#include "host_datatypes.h"

#ifdef __cplusplus
extern "C"
{
#endif
AcResult acTransposeWithBounds(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream);
AcResult acTranspose(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const cudaStream_t stream);
#ifdef __cplusplus
}
#endif
