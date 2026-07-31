#pragma once

/*
 * Random number generation
 */

#include <cstdint>

#if AC_CPU_BUILD
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

#if AC_CPU_BUILD

__device__ __forceinline__ AcReal rand_uniform();

#else

typedef curandStateXORWOW_t acRandState;
extern __device__ __constant__ acRandState* rand_states;

static __global__ void
rand_init(const uint64_t seed, const size_t count, const size_t rank);

#if AC_DOUBLE_PRECISION
#define rand_uniform() curand_uniform_double(&rand_states[local_compdomain_idx])
#else
#define rand_uniform() curand_uniform(&rand_states[local_compdomain_idx])
#endif

__device__ __forceinline__ AcReal random_uniform(const size_t idx);

#endif

AC_END_C_DECLARATIONS
