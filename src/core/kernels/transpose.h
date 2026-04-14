#pragma once
#include "func_define.h"
#include "device_headers.h"
#include "acreal.h"
#include "host_datatypes.h"

AC_BEGIN_C_DECLARATIONS

AcResult acTransposeWithBounds(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream);
AcResult acTranspose(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const cudaStream_t stream);

AC_END_C_DECLARATIONS
