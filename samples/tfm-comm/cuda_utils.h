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
#endif

#if defined(DEVICE_ENABLED)
static inline cudaStream_t*
cuda_stream_create(const unsigned int flags = cudaStreamDefault)
{
    PRINT_LOG("new stream");
    cudaStream_t* stream = new cudaStream_t;
    // ERRCHK_CUDA_API(cudaStreamCreate(stream));
    ERRCHK_CUDA_API(cudaStreamCreateWithFlags(stream, flags));
    return stream;
}

static inline void
cuda_stream_destroy(cudaStream_t* stream) noexcept
{
    PRINT_LOG("delete stream");
    WARNCHK_CUDA_API(cudaStreamDestroy(*stream));
    delete stream;
}
#else
using cudaStream_t                       = unsigned int;
constexpr unsigned int cudaStreamDefault = 0;

static inline cudaStream_t*
cuda_stream_create(const unsigned int flags = cudaStreamDefault)
{
    PRINT_LOG("new stream");
    cudaStream_t* stream = new cudaStream_t;
    *stream              = flags;
    return stream;
}

static inline void
cuda_stream_destroy(cudaStream_t* stream) noexcept
{
    delete stream;
}
#endif
