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

enum BufferType {
    BUFFER_HOST,
    BUFFER_HOST_PINNED,
    BUFFER_HOST_PINNED_WRITE_COMBINED,
    BUFFER_DEVICE,
    BUFFER_NULL,
};

template <typename T> struct Buffer {
    size_t count;
    BufferType type;
    T* data;

    // Constructors
    Buffer()
        : count(0), type(BUFFER_NULL), data(nullptr) {};
    Buffer(const size_t in_count, const BufferType in_type = BUFFER_HOST);
    Buffer(const Buffer&)            = delete; // Copy
    Buffer& operator=(const Buffer&) = delete; // Copy assignment
    Buffer(Buffer&&) noexcept;                 // Move
    Buffer& operator=(Buffer&&);               // Move assignment
    ~Buffer();

    // Member functions
    void fill(const T& value);
    void fill_arange(const T& min, const T& max);
    void migrate(Buffer<T>& out) const;
};

template <typename T>
Buffer<T>::Buffer(const size_t in_count, const BufferType in_type)
    : count(in_count), type(in_type), data(nullptr)
{
#if !defined(DEVICE_ENABLED)
    if (in_type != BUFFER_HOST) {
        WARNING_DESC("Device code not enabled, falling back to host allocation");
        type = BUFFER_HOST;
    }
#endif

    // Allocate page-locked host memory
    // - cudaHostAllocDefault: emulates to cudaMallocHost (allocates page-locked memory)
    // - cudaHostAllocPortable: memory considered pinned by all CUDA contexts
    // - cudaHostAllocMapped: allocates a host buffer the device can access directly,
    // generates implicit PCI-e traffic. Likely used by unified memory cudaMallocManaged
    // under the hood (See CUDA C programming guide 6.2.6.3)
    // - cudaHostAllocWriteCombined: bypasses host L1/L2 to improve host-device transfers
    // but results in very slow host-side reads (See CUDA C programming guide 6.2.6.2)
    // unsigned int flags = cudaHostAllocDefault;
    if (type == BUFFER_HOST) {
        data = new T[count];
    }
#if defined(DEVICE_ENABLED)
    else if (type == BUFFER_HOST_PINNED) {
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocDefault));
    }
    else if (type == BUFFER_HOST_PINNED_WRITE_COMBINED) {
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocWriteCombined));
    }
    else if (type == BUFFER_DEVICE) {
        ERRCHK_CUDA_API(cudaMalloc(&data, count * sizeof(data[0])));
    }
#endif
    else {
        ERROR_DESC("Invalid type");
    }
}

template <typename T>
Buffer<T>::Buffer(Buffer&& other) noexcept
    : count(other.count), type(other.type), data(other.data)
{
    other.count = 0;
    other.type  = BUFFER_NULL;
    other.data  = nullptr;
}

template <typename T>
Buffer<T>&
Buffer<T>::operator=(Buffer&& other)
{
    if (this != &other) {
        if (type == BUFFER_HOST) {
            delete[] data;
        }
#if defined(DEVICE_ENABLED)
        else if (type == BUFFER_HOST_PINNED || type == BUFFER_HOST_PINNED_WRITE_COMBINED) {
            WARNCHK_CUDA_API(cudaFreeHost(data));
        }
        else if (type == BUFFER_DEVICE) {
            WARNCHK_CUDA_API(cudaFree(data));
        }
#endif
        count       = other.count;
        type        = other.type;
        data        = other.data;
        other.count = 0;
        other.type  = BUFFER_NULL;
        other.data  = nullptr;
    }
    return *this;
}

template <typename T> Buffer<T>::~Buffer()
{
    if (type == BUFFER_HOST) {
        delete[] data;
    }
#if defined(DEVICE_ENABLED)
    else if (type == BUFFER_HOST_PINNED || type == BUFFER_HOST_PINNED_WRITE_COMBINED) {
        WARNCHK_CUDA_API(cudaFreeHost(data));
    }
    else if (type == BUFFER_DEVICE) {
        WARNCHK_CUDA_API(cudaFree(data));
    }
#endif
    else {
        WARNING_DESC("Invalid type");
    }
    data = nullptr;
}

/**
 * Printing
 */
template <typename T>
std::ostream&
operator<<(std::ostream& os, const Buffer<T>& buf)
{
    ERRCHK(buf.type != BUFFER_DEVICE); // TODO implement for device buffers
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
Buffer<T>::fill(const T& value)
{
    ERRCHK(type != BUFFER_DEVICE); // TODO implement for device buffers
    for (size_t i = 0; i < count; ++i)
        data[i] = value;
}

template <typename T>
void
Buffer<T>::fill_arange(const T& min, const T& max)
{
    ERRCHK(type != BUFFER_DEVICE); // TODO implement for device buffers
    ERRCHK(min < max);
    ERRCHK(max - min <= count);
    for (size_t i = 0; i < max - min; ++i)
        data[i] = static_cast<T>(min + i);
}

template <typename T>
void
Buffer<T>::migrate(Buffer<T>& out) const
{
    ERRCHK(count == out.count);
    ERRCHK(type != BUFFER_NULL);
    if (type == BUFFER_HOST && out.type == BUFFER_HOST) {
        std::copy(data, data + count, out.data);
    }
    else {
#if defined(DEVICE_ENABLED)
        const bool on_device     = (type == BUFFER_DEVICE);
        const bool out_on_device = (out.type == BUFFER_DEVICE);
        cudaMemcpyKind kind;
        if (on_device) {
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
        ERRCHK_CUDA_API(cudaMemcpy(out.data, data, sizeof(data[0]) * count, kind));
#else
        ERROR_DESC("invalid type");
#endif
    }
}

/**
 * Other utility functions
 */
template <typename T>
Buffer<T>
ones(const size_t count)
{
    Buffer<T> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T>
zeros(const size_t count)
{
    Buffer<T> buffer(count);
    for (size_t i = 0; i < count; ++i)
        buffer.data[i] = 1;
    return buffer;
}

template <typename T>
Buffer<T>
arange(const size_t min, const size_t max)
{
    ERRCHK(max > min);
    Buffer<T> buffer(max - min);
    for (size_t i = 0; i < buffer.count; ++i)
        buffer.data[i] = static_cast<T>(min + i);
    return buffer;
}

/**
 * Testing
 */
void test_buffer();
