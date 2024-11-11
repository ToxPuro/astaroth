#pragma once
#include <cstddef>
#include <memory>

#include "errchk.h"
#include "print_debug.h"

struct HostMemoryResource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("host");
        void* ptr = malloc(bytes);
        ERRCHK(ptr);
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG("host");
        WARNCHK(ptr);
        free(ptr);
    }
};

#if defined(DEVICE_ENABLED)
#if defined(CUDA_ENABLED)
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#endif

struct PinnedHostMemoryResource : public HostMemoryResource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("host pinned");
        void* ptr = nullptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG("host pinned");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

struct PinnedWriteCombinedHostMemoryResource : public HostMemoryResource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("host pinned write-combined");
        void* ptr = nullptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG("host pinned write-combined");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

struct DeviceMemoryResource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("device");
        void* ptr = nullptr;
        ERRCHK_CUDA_API(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG("device");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFree(ptr));
    }
};
#else
#pragma message("Device code was not enabled. Falling back to host-only memory allocations")
using PinnedHostMemoryResource              = HostMemoryResource;
using PinnedWriteCombinedHostMemoryResource = HostMemoryResource;
using DeviceMemoryResource                  = HostMemoryResource;
#endif
