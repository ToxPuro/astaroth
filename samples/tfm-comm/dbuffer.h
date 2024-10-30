#pragma once
#include <stddef.h>

#if defined(__CUDA_ARCH__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(__HIP_DEVICE_COMPILE__)
#define DEVICE_ENABLED
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
#endif

enum DeviceBufferType {
    DEVICE_BUFFER_HOST,
    DEVICE_BUFFER_HOST_PINNED,
    DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED,
    DEVICE_BUFFER_DEVICE,
};

template <typename T> struct DeviceBuffer {
    const size_t count;
    const DeviceBufferType type;
    T* data;

    DeviceBuffer(const size_t count, const DeviceBufferType type = DEVICE_BUFFER_HOST);

    ~DeviceBuffer();

    // Delete all other types of constructors
    DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
    DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
    DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator
};

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const size_t count, const DeviceBufferType type)
    : count(count), type(type), data(nullptr)
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
    case DEVICE_BUFFER_HOST:
        data = new T[count];
        break;
#if defined(DEVICE_ENABLED)
    case DEVICE_BUFFER_HOST_PINNED:
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocDefault));
        break;
    case DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED:
        ERRCHK_CUDA_API(cudaHostAlloc(&data, count * sizeof(data[0]), cudaHostAllocWriteCombined));
        break;
    case DEVICE_BUFFER_DEVICE:
        ERRCHK_CUDA_API(cudaMalloc(&data, count * sizeof(data[0])));
        break;
#endif
    default:
        WARNING_DESC("Device code not enabled, falling back to host allocation");
        data = new T[count];
    }
}

template <typename T> DeviceBuffer<T>::~DeviceBuffer()
{
    switch (type) {
    case DEVICE_BUFFER_HOST:
        delete[] data;
        break;
#if defined(DEVICE_ENABLED)
    case DEVICE_BUFFER_HOST_PINNED: /* Fallthrough*/
    case DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED:
        WARNCHK_CUDA_API(cudaFreeHost(data));
        break;
    case DEVICE_BUFFER_DEVICE:
        WARNCHK_CUDA_API(cudaFree(data));
        break;
#endif
    default:
        WARNING_DESC("Device code not enabled, falling back to host free");
        delete[] data;
    }
    data = nullptr;
}
