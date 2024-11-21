#pragma once
#include <cstddef>
#include <memory>

#include "errchk.h"
#include "print_debug.h"

namespace ac::mr {

struct host_memory_resource {
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

} // namespace ac::mr

#if defined(DEVICE_ENABLED)

#if defined(CUDA_ENABLED)
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "hip.h"
#include <hip/hip_runtime.h>
#endif

#include "errchk_cuda.h"

namespace ac::mr {

struct pinned_host_memory_resource : public host_memory_resource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("host pinned");
        void* ptr{nullptr};
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

struct pinned_write_combined_host_memory_resource : public host_memory_resource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("host pinned write-combined");
        void* ptr{nullptr};
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

struct device_memory_resource {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG("device");
        void* ptr{nullptr};
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

} // namespace ac::mr

#else

#pragma message("Device code was not enabled. Falling back to host-only memory allocations")
namespace ac::mr {
using pinned_host_memory_resource                = ac::mr::host_memory_resource;
using pinned_write_combined_host_memory_resource = ac::mr::host_memory_resource;
using device_memory_resource                     = ac::mr::host_memory_resource;
} // namespace ac::mr

#endif
