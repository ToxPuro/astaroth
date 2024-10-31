#pragma once
#include <stddef.h>

#include <iostream> // Printing

#if defined(__CUDA_ARCH__)
#define ENABLED
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(__HIP_COMPILE__)
#define ENABLED
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

    Buffer(const size_t count, const BufferType type = BUFFER_HOST);
    ~Buffer();
    Buffer(Buffer&& other) noexcept; // Move constructor

    // Delete all other types of constructors
    Buffer(const Buffer&)            = delete; // Copy constructor
    Buffer& operator=(const Buffer&) = delete; // Copy assignment operator
    // Buffer(Buffer&&)                 = delete; // Move constructor
    Buffer& operator=(Buffer&&) = delete; // Move assignment operator

    // Other functions
    void fill(const T& value);
    void fill_arange(const T& min, const T& max);
};

template <typename T>
Buffer<T>::Buffer(const size_t count_, const BufferType type_)
    : count(count_), type(type_), data(nullptr)
{
    // Allocate page-locked host memory
    // - cudaHostAllocDefault: emulates to cudaMallocHost (allocates page-locked memory)
    // - cudaHostAllocPortable: memory considered pinned by all CUDA contexts
    // - cudaHostAllocMapped: allocates a host buffer the device can access directly,
    // generates implicit PCI-e traffic. Likely used by unified memory cudaMallocManaged
    // under the hood (See CUDA C programming guide 6.2.6.3)
    // - cudaHostAllocWriteCombined: bypasses host L1/L2 to improve host-device transfers
    // but results in very slow host-side reads (See CUDA C programming guide 6.2.6.2)
    // unsigned int flags = cudaHostAllocDefault;
    switch (type) {
    case BUFFER_HOST:
        data = new T[count];
        break;
#if defined(ENABLED)
    case BUFFER_HOST_PINNED:
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocDefault));
        break;
    case BUFFER_HOST_PINNED_WRITE_COMBINED:
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocWriteCombined));
        break;
    case BUFFER_DEVICE:
        ERRCHK_CUDA_API(cudaMalloc(&data, count * sizeof(data[0])));
        break;
#endif
    default:
        WARNING_DESC("Device code not enabled, falling back to host allocation");
        type = BUFFER_HOST;
        data = new T[count];
    }
}

template <typename T> Buffer<T>::~Buffer()
{
    switch (type) {
    case BUFFER_HOST:
        delete[] data;
        break;
#if defined(ENABLED)
    case BUFFER_HOST_PINNED: /* Fallthrough*/
    case BUFFER_HOST_PINNED_WRITE_COMBINED:
        WARNCHK_CUDA_API(cudaFreeHost(data));
        break;
    case BUFFER_DEVICE:
        WARNCHK_CUDA_API(cudaFree(data));
        break;
#endif
    default:
        WARNING_DESC("Invalid type");
    }
    data = nullptr;
}

template <typename T>
Buffer<T>::Buffer(Buffer&& other) noexcept
    : count(other.count), type(other.type), data(other.data)
{
    other.count = 0;
    other.type  = BUFFER_NULL;
    other.data  = nullptr;
}

/**
 * Printing
 */
template <typename T>
std::ostream&
operator<<(std::ostream& os, const Buffer<T>& buf)
{
    if (buf.type != BUFFER_DEVICE) {
        os << "{";
        for (size_t i = 0; i < buf.count; ++i)
            os << buf.data[i] << (i + 1 < buf.count ? ", " : "}");
        return os;
    }
    else {
        os << "{ Device buffer. Printing not implemented. }";
        return os;
    }
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
