#pragma once

// Remove __host__ __device__ if compiling without device libraries
#if defined(ACM_DEVICE_ENABLED)

#if defined(ACM_CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(ACM_HIP_ENABLED)
#include "hip.h"
// #pragma gcc system_header // TODO check that this does not disable warnings elsewhere
#pragma clang system_header // TODO check that this does not disable warnings elsewhere
#include <hip/hip_runtime.h>
#else
#if !defined(ACM_HOST_ONLY_MODE_ENABLED)
static_assert(false,
              "Device code was enabled but neither ACM_CUDA_ENABLED nor ACM_HIP_ENABLED is set");
#endif
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
