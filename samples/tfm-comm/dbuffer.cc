#include "dbuffer.h"

#include "errchk_cuda.h"

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
    case DEVICE_BUFFER_HOST_PINNED:
        ERRCHK_CUDA_API(cudaAllocHost(&data, count * sizeof(data[0]), cudaHostAllocDefault));
        break;
    case DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED:
        ERRCHK_CUDA_API(cudaAllocHost(&data, count * sizeof(data[0]), cudaHostAllocWriteCombined));
        break;
    case DEVICE_BUFFER_DEVICE:
        ERRCHK_CUDA_API(cudaMalloc(&data, count * sizeof(data[0])));
        break;
    }
}

template <typename T> DeviceBuffer<T>::~DeviceBuffer()
{
    switch (type) {
    case DEVICE_BUFFER_HOST:
        delete[] data;
        break;
    case DEVICE_BUFFER_HOST_PINNED: /* Fallthrough*/
    case DEVICE_BUFFER_HOST_PINNED_WRITE_COMBINED:
        ERRCHK_CUDA_API(cudaFreeHost(data));
        break;
    case DEVICE_BUFFER_DEVICE:
        ERRCHK_CUDA_API(cudaFree(data));
        break;
    }
}
