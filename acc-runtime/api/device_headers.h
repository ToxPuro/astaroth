#pragma once
#pragma GCC system_header // NOTE: Silences errors originating from CUDA/HIP
                          // headers

#if AC_USE_HIP
#include "hip.h"

#include <hip/hip_runtime_api.h>     // Streams
#if PROFILING_ENABLED
#include <roctracer/roctracer_ext.h> // Profiling
#endif
#else
#if PROFILING_ENABLED
#include <cuda_profiler_api.h> // Profiling
#endif
#include <cuda_runtime_api.h>  // Streams
#endif
