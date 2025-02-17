#pragma once
#pragma GCC system_header // NOTE: Silences errors originating from CUDA/HIP
                          // headers

#if AC_USE_HIP
#include "hip.h"

#include <hip/hip_runtime_api.h>     // Streams
#include <roctracer/roctracer_ext.h> // Profiling
#else
#include <cuda_profiler_api.h> // Profiling
#include <cuda_runtime_api.h>  // Streams
#endif
