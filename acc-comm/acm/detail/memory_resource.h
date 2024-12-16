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

#if defined(ACM_DEVICE_ENABLED)

#include "cuda_utils.h"
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

namespace ac::mr {

template <typename MemoryResourceA, typename MemoryResourceB>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceA>) {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
            PRINT_LOG("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<ac::mr::device_memory_resource, MemoryResourceB>) {
            PRINT_LOG("htod");
            return cudaMemcpyHostToDevice;
        }
        else {
            PRINT_LOG("htoh");
            return cudaMemcpyHostToHost;
        }
    }
}

template <typename T, typename MemoryResource> class base_ptr {
  protected:
    const size_t _size;
    T* _data;

  public:
    explicit base_ptr(const size_t size, T* data)
        : _size{size}, _data{data}
    {
    }

    size_t size() const { return _size; }
    T* data() { return _data; }
    T* data() const { return _data; }

    // Enable the subscript[] operator
    T& operator[](const size_t i)
    {
        ERRCHK(i < _size);
        return _data[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < _size);
        return _data[i];
    }
};

template <typename T> using host_ptr   = base_ptr<T, ac::mr::host_memory_resource>;
template <typename T> using device_ptr = base_ptr<T, ac::mr::device_memory_resource>;

#if defined(ACM_DEVICE_ENABLED)

template <typename T, typename MemoryResourceA, typename MemoryResourceB>
void
copy(const base_ptr<T, MemoryResourceA> in, base_ptr<T, MemoryResourceB> out)
{
    ERRCHK(in.size() <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(T),
                               ac::mr::get_kind<MemoryResourceA, MemoryResourceB>()));
}

#else

template <typename T> using device_ptr = host_ptr<T>;

template <typename T>
void
copy(const host_ptr<T>& in, host_ptr<T>& out)
{
    ERRCHK(in.size() <= out.size());
    std::copy(in.data(), in.data() + in.size(), out.data());
}

#endif

} // namespace ac::mr

void test_memory_resource();
