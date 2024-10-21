#pragma once

#include "errchk.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
static_assert(false);
#endif

static inline void
errchk_print_cuda_api_error(const int errorcode, const char* function, const char* file,
                            const long line, const char* expression)
{
    errchk_print_error(function, file, line, expression, cudaGetErrorString(errorcode));
}

#define ERRCHK_CUDA_API(errcode)                                                                   \
    do {                                                                                           \
        const int errchk_cuda_api_code__ = (errcode);                                              \
        if (errchk_cuda_api_code__ != cudaSuccess) {                                               \
            print_cuda_api_error(errchk_cuda_api_code__, __func__, __FILE__, __LINE__, #errcode);  \
            throw std::runtime_error("CUDA API error");                                            \
        }                                                                                          \
    } while (0)

#define ERRCHK_CUDA_KERNEL(errcode)                                                                \
    do {                                                                                           \
        ERRCHK_CUDA_API(cudaPeekAtLastError());                                                    \
        ERRCHK_CUDA_API(cudaDeviceSynchronize());                                                  \
    } while (0)
