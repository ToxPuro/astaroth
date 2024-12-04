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

template <typename T> class base_ptr {
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
};

template <typename T> class host_ptr : public base_ptr<T> {
  public:
    using base_ptr<T>::base_ptr;
};

#if defined(ACM_DEVICE_ENABLED)
template <typename T> class device_ptr : public base_ptr<T> {
  public:
    using base_ptr<T>::base_ptr;
};

template <typename T>
void
copy(const base_ptr<T>& in, const cudaMemcpyKind& kind, base_ptr<T>& out)
{
    ERRCHK(in.size() <= out.size());
    ERRCHK_CUDA_API(cudaMemcpy(out.data(), in.data(), in.size() * sizeof(T), kind));
}
template <typename T>
void
copy(const device_ptr<T>& in, device_ptr<T>& out)
{
    ac::mr::copy(in, cudaMemcpyDeviceToDevice, out);
}
template <typename T>
void
copy(const device_ptr<T>& in, host_ptr<T>& out)
{
    ac::mr::copy(in, cudaMemcpyDeviceToHost, out);
}
template <typename T>
void
copy(const host_ptr<T>& in, device_ptr<T>& out)
{
    ac::mr::copy(in, cudaMemcpyHostToDevice, out);
}
template <typename T>
void
copy(const host_ptr<T>& in, host_ptr<T>& out)
{
    ac::mr::copy(in, cudaMemcpyHostToHost, out);
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
