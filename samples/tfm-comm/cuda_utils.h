#pragma once

#include <memory.h>

#if defined(__CUDACC__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
using cudaStream_t                       = void*;
constexpr unsigned int cudaStreamDefault = 0;
#endif

cudaStream_t* cuda_stream_create(const unsigned int flags = cudaStreamDefault);

void cuda_stream_destroy(cudaStream_t* stream) noexcept;
