#pragma once

// Remove __host__ __device__ if compiling without device libraries
#if defined(DEVICE_ENABLED)

#if defined(CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "hip.h"
// #pragma gcc system_header // TODO check that this does not disable warnings elsewhere
#pragma clang system_header // TODO check that this does not disable warnings elsewhere
#include <hip/hip_runtime.h>
#else
static_assert(false, "Device code was enabled but neither CUDA_ENABLED nor HIP_ENABLED is set");
#endif

#else
#define __host__
#define __device__
#endif

// Disable errchecks in device code (not supported as of 2024-11-11)
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1)
#undef ERRCHK
#define ERRCHK(expr)
#undef ERRCHK_EXPR_DESC
#define ERRCHK_EXPR_DESC(expr, ...)
#endif
