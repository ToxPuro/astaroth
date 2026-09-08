#pragma once

#include "acc_runtime.h"
#include "acreal.h"
#include "astaroth_device_headers.h"
#include "astaroth_helpers.h"
#include "func_define.h"
#include "host_datatypes.h"
#include "ac_reduce_helpers.h"

AC_BEGIN_C_DECLARATIONS

void
ac_unset_floating_point_exceptions();

void
ac_restore_floating_point_exceptions();

size_t
acShapeCount(const AcShape shape);

int
acGetNumOfWarps(const dim3 bpg, const dim3 tpb);

int
acGetCurrentDevice();

bool
acSupportsCooperativeLaunches();


size_t acGetSizeFromDim(const int dim, const Volume dims);

Volume acGetVolumeFromShape(const AcShape shape);
int acMemUsage();

size_t
acGetAmountOfDeviceMemoryFree();

size_t
acDeviceResize(void** dst,const size_t old_bytes,const size_t new_bytes);

Volume
get_bpg(Volume dims, const Volume tpb);

AcShape  acGetTransposeBufferShape(const AcMeshOrder order, const Volume dims);
AcShape  acGetReductionShape(const AcProfileType type, const AcMeshDims dims);
AcMeshOrder acGetMeshOrderForProfile(const AcProfileType type);

// Returns the number of elements contained within shape
size_t acShapeSize(const AcShape shape);

AC_END_C_DECLARATIONS

#ifdef __cplusplus

cudaDeviceProp get_device_prop();

int3
ceil(AcReal3 a);

size3_t
ceil_div(const size3_t& a, const int3& b);

size3_t
ceil_div(const size3_t& a, const size3_t& b);

int3
ceil_div(const int3& a, const int3& b);

size_t
ceil_div(const size_t& a, const size_t& b); 

size3_t
ceil_div(const size3_t& a, const int& b); 

void
acDeviceMalloc(void** dst, const size_t bytes);
void
acDeviceMalloc(AcReal** dst, const size_t bytes);

void
acDeviceFree(void** dst, const int bytes);
void
acDeviceFree(AcReal** dst, const int bytes);
void
acDeviceFree(AcComplex** dst, const int bytes);

#endif
