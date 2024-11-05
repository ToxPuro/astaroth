#pragma once
#include <cstddef>
#include <memory>

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

#include "print_debug.h"

template <typename T> struct HostMemoryResource {
    static T* nalloc(const size_t count)
    {
        PRINT_LOG("host");
        T* ptr = (T*)malloc(count * sizeof(ptr[0]));
        ERRCHK(ptr);
        return ptr;
    }

    static void ndealloc(T* ptr) noexcept
    {
        PRINT_LOG("host");
        WARNCHK(ptr);
        free(ptr);
    }
};

#if defined(DEVICE_ENABLED)
template <typename T> struct PinnedHostMemoryResource : public HostMemoryResource<T> {
    static T* nalloc(const size_t count)
    {
        PRINT_LOG("host pinned");
        T* ptr = nullptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, count * sizeof(ptr[0]), cudaHostAllocDefault));
        return ptr;
    }

    static void ndealloc(T* ptr) noexcept
    {
        PRINT_LOG("host pinned");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

template <typename T> struct PinnedWriteCombinedHostMemoryResource : public HostMemoryResource<T> {
    static T* nalloc(const size_t count)
    {
        PRINT_LOG("host pinned write-combined");
        T* ptr = nullptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, count * sizeof(ptr[0]), cudaHostAllocWriteCombined));
        return ptr;
    }

    static void ndealloc(T* ptr) noexcept
    {
        PRINT_LOG("host pinned write-combined");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

template <typename T> struct DeviceMemoryResource {
    static T* nalloc(const size_t count)
    {
        PRINT_LOG("device");
        T* ptr = nullptr;
        ERRCHK_CUDA_API(cudaMalloc(&ptr, count * sizeof(ptr[0])));
        return ptr;
    }

    static void ndealloc(T* ptr) noexcept
    {
        PRINT_LOG("device");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFree(ptr));
    }
};
#else
#pragma message("Device code was not enabled. Falling back to host-only memory allocations")
template <typename T> using PinnedHostMemoryResource              = HostMemoryResource<T>;
template <typename T> using PinnedWriteCombinedHostMemoryResource = HostMemoryResource<T>;
template <typename T> using DeviceMemoryResource                  = HostMemoryResource<T>;
#endif
