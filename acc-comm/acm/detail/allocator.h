#pragma once
#include <cstddef>

#include "errchk.h"

#if defined(ACM_DEVICE_ENABLED)
#include "cuda_utils.h"
#include "errchk_cuda.h"
#endif

namespace ac::mr {

struct host_allocator {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG_TRACE("host");
        void* ptr = malloc(bytes);
        ERRCHK(ptr);
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG_TRACE("host");
        WARNCHK(ptr);
        free(ptr);
    }
};

#if defined(ACM_DEVICE_ENABLED)

struct pinned_host_allocator : public host_allocator {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG_TRACE("host pinned");
        void* ptr{nullptr};
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG_TRACE("host pinned");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

struct pinned_write_combined_host_allocator : public host_allocator {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG_TRACE("host pinned write-combined");
        void* ptr{nullptr};
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, bytes, cudaHostAllocWriteCombined));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG_TRACE("host pinned write-combined");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFreeHost(ptr));
    }
};

struct device_allocator {
    static void* alloc(const size_t bytes)
    {
        PRINT_LOG_TRACE("device");
        void* ptr{nullptr};
        ERRCHK_CUDA_API(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    static void dealloc(void* ptr) noexcept
    {
        PRINT_LOG_TRACE("device");
        WARNCHK(ptr);
        WARNCHK_CUDA_API(cudaFree(ptr));
    }
};

#else

#if !defined(ACM_HOST_ONLY_MODE_ENABLED)
#pragma message("Device code was not enabled. Falling back to host-only memory allocations")
#endif

using pinned_host_allocator                = host_allocator;
using pinned_write_combined_host_allocator = host_allocator;
using device_allocator                     = host_allocator;

#endif

} // namespace ac::mr

void test_allocator();
