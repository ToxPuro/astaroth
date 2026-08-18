#pragma once

/*
 * Random number generation
 */

#include <cstdint>

#if AC_CPU_BUILD
#include <random>

#include <stdlib.h>
#else
#if AC_USE_HIP
#include <hip/hip_fp16.h>            // Workaround: required by hiprand
#include <hiprand/hiprand.h>         // Random numbers
#include <hiprand/hiprand_kernel.h>  // Random numbers (device)
#else
#include <curand.h>         // Random numbers
#include <curand_kernel.h>  // Random numbers (device)
#endif
#endif

// clang-format off
#include "host_datatypes.h"
// clang-format on

#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "func_define.h"

AC_BEGIN_C_DECLARATIONS

AcResult acRandInitAlt(const uint64_t, const size_t, const size_t);
void acRandQuit(void);

#if AC_CPU_BUILD

__device__ __forceinline__ AcReal rand_uniform();

#else

typedef curandStateXORWOW_t acRandState;

extern __device__ __constant__ acRandState* rand_states;

#if AC_DOUBLE_PRECISION
#define rand_uniform() curand_uniform_double(&rand_states[local_compdomain_idx])
#else
#define rand_uniform() curand_uniform(&rand_states[local_compdomain_idx])
#endif /* AC_DOUBLE_PRECISION */

#endif /* AC_CPU_BUILD */

AC_END_C_DECLARATIONS
