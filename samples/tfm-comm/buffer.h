#pragma once
#include <stddef.h>

#include <iostream> // Printing

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

// #define DEVICE_ENABLED

struct HostAllocator {
    static void* alloc(const size_t count, const size_t size)
    {
        WARNING_DESC("host");
        void* ptr = malloc(count * size);
        ERRCHK(ptr);
        return ptr;
    }
    static void dealloc(void** ptr)
    {
        WARNCHK(ptr);
        WARNCHK(*ptr);
        free(*ptr);
        *ptr = NULL;
    }
};

#if defined(DEVICE_ENABLED)
struct DeviceAllocator {
    static void* alloc(const size_t count, const size_t size)
    {
        WARNING_DESC("device");
        void* ptr;
        ERRCHK_CUDA_API(cudaMalloc(&ptr, count * size));
        return ptr;
    }
    static void dealloc(void** ptr)
    {
        WARNING_DESC("device freed");
        WARNCHK(ptr);
        WARNCHK(*ptr);
        WARNCHK_CUDA_API(cudaFree(*ptr));
        *ptr = NULL;
    }
};

struct HostAllocatorPinned {
    static void* alloc(const size_t count, const size_t size)
    {
        WARNING_DESC("host pinned");
        void* ptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, count * size, cudaHostAllocDefault));
        return ptr;
    }
    static void dealloc(void** ptr)
    {
        WARNCHK(ptr);
        WARNCHK(*ptr);
        WARNCHK_CUDA_API(cudaFreeHost(*ptr));
        *ptr = NULL;
    }
};

struct HostAllocatorPinnedWriteCombined {
    static void* alloc(const size_t count, const size_t size)
    {
        WARNING_DESC("host pinned write-combined");
        void* ptr;
        ERRCHK_CUDA_API(cudaHostAlloc(&ptr, count * size, cudaHostAllocWriteCombined));
        return ptr;
    }
    static void dealloc(void** ptr)
    {
        WARNCHK(ptr);
        WARNCHK(*ptr);
        WARNCHK_CUDA_API(cudaFreeHost(*ptr));
        *ptr = NULL;
    }
};
#else
#pragma message "Buffer: Device code not enabled, reverting back to host allocator"
using DeviceAllocator                  = HostAllocator;
using HostAllocatorPinned              = HostAllocator;
using HostAllocatorPinnedWriteCombined = HostAllocator;
#endif

template <typename T, typename Allocator = HostAllocator> struct Buffer {
    size_t count;
    T* data;

    Buffer(const size_t in_count)
        : count(in_count), data((T*)Allocator::alloc(count, sizeof(data[0])))
    {
    }
    ~Buffer()
    {
        Allocator::dealloc((void**)&data);
        count = 0;
    }
    // template <typename OtherAllocator> void migrate(Buffer<T, OtherAllocator> other) const
    // {
    //     ERRCHK(count == other.count);
    //     if (std::is_same<Allocator, HostAllocator>::value &&
    //         std::is_same<OtherAllocator, HostAllocator>::value) {
    //         std::copy(data.begin(), data.end(), other.data.begin());
    //     }
    //     else if (std::is_same<Allocator, HostAllocatorPinned>::value ||
    //              std::is_same<Allocator, HostAllocatorPinnedWriteCombined>::value) {
    //         if (std::is_same<OtherAllocator, HostAllocatorPinned>::value ||
    //             std::is_same<OtherAllocator, HostAllocatorPinnedWriteCombined>::value) {
    //             ERRCHK_CUDA_API(
    //                 cudaMemcpy(other.data, data, sizeof(data[0]) * count, cudaMemcpyHostToHost));
    //         }
    //         else if (std::is_same<OtherAllocator, DeviceAllocator>::value) {
    //             ERRCHK_CUDA_API(
    //                 cudaMemcpy(other.data, data, sizeof(data[0]) * count,
    //                 cudaMemcpyHostToDevice));
    //         }
    //         else {
    //             static_assert(false);
    //         }
    //     }
    //     else {
    //         if (std::is_same<OtherAllocator, HostAllocatorPinned>::value ||
    //             std::is_same<OtherAllocator, HostAllocatorPinnedWriteCombined>::value) {
    //             ERRCHK_CUDA_API(
    //                 cudaMemcpy(other.data, data, sizeof(data[0]) * count,
    //                 cudaMemcpyDeviceToHost));
    //         }
    //         else if (std::is_same<OtherAllocator, DeviceAllocator>::value) {
    //             ERRCHK_CUDA_API(cudaMemcpy(other.data, data, sizeof(data[0]) * count,
    //                                           cudaMemcpyDeviceToDevice));
    //         }
    //         else {
    //             static_assert(false);
    //         }
    //     }
    // }

    // Delete all other types of constructors
    Buffer(const Buffer&)            = delete; // Copy constructor
    Buffer& operator=(const Buffer&) = delete; // Copy assignment operator
    // Buffer(Buffer&&)                 = delete; // Move constructor
    Buffer& operator=(Buffer&&) = delete; // Move assignment operator

    Buffer(Buffer&& other) noexcept
        : count(other.count), data(other.data)
    {
        other.count = 0;
        other.data  = nullptr;
    }
};

/**
 * Printing
 */
template <typename T>
std::ostream&
operator<<(std::ostream& os, const Buffer<T, HostAllocator>& buf)
{
    os << "{";
    for (size_t i = 0; i < buf.count; ++i)
        os << buf.data[i] << (i + 1 < buf.count ? ", " : "}");
    return os;
}

/**
 * Member functions
 */
template <typename T>
void
migrate(const Buffer<T, HostAllocator>& in, Buffer<T, HostAllocator>& out)
{
    std::cerr << "default copy" << std::endl;
    ERRCHK(in.count == out.count);
    std::copy(in.data, in.data + in.count, out.data);
}

#if defined(DEVICE_ENABLED)
template <typename T, typename A, typename B>
void
migrate(const Buffer<T, A>& in, Buffer<T, B>& out)
{
    ERRCHK(in.count == out.count);
    const bool in_on_device  = std::is_same<A, DeviceAllocator>::value;
    const bool out_on_device = std::is_same<B, DeviceAllocator>::value;

    cudaMemcpyKind kind;
    if (in_on_device) {
        if (out_on_device)
            kind = cudaMemcpyDeviceToDevice;
        else
            kind = cudaMemcpyDeviceToHost;
    }
    else {
        if (out_on_device)
            kind = cudaMemcpyHostToDevice;
        else
            kind = cudaMemcpyHostToHost;
    }
    std::cerr << "kind " << (kind == cudaMemcpyDeviceToHost) << std::endl;
    ERRCHK_CUDA_API(cudaMemcpy(out.data, in.data, sizeof(in.data[0]) * in.count, kind));
}
#endif

template <typename T>
void
fill(const T& value, Buffer<T, HostAllocator>& buffer)
{
    for (size_t i = 0; i < buffer.count; ++i)
        buffer.data[i] = value;
}

template <typename T>
void
fill_arange(const T& min, const T& max, Buffer<T, HostAllocator>& buffer)
{
    ERRCHK(min < max);
    ERRCHK(max - min <= buffer.count);
    for (size_t i = 0; i < max - min; ++i)
        buffer.data[i] = static_cast<T>(min + i);
}

/**
 * Other utility functions
 */
template <typename T>
Buffer<T, HostAllocator>
ones(const size_t count)
{
    Buffer<T, HostAllocator> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T, HostAllocator>
zeros(const size_t count)
{
    Buffer<T, HostAllocator> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T, HostAllocator>
arange(const size_t min, const size_t max)
{
    ERRCHK(max > min);
    Buffer<T, HostAllocator> buffer(max - min);
    for (size_t i = 0; i < buffer.count; ++i)
        buffer.data[i] = static_cast<T>(min + i);
    return buffer;
}

/**
 * Testing
 */
void test_buffer();
