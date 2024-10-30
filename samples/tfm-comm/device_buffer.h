#pragma once

#include <iostream>
#include <memory>

#include "errchk.h"

template <typename T> struct DeviceBuffer {
    size_t count;
    bool on_device;
    T* data;

    // Constructor
    DeviceBuffer(const size_t count = 0, const bool on_device = false,
                 unsigned int flags = cudaHostAllocDefault)
        : count(count), on_device(on_device), data(nullptr)
    {
        if (on_device) {
            ERRCHK_CUDA_API(cudaMalloc(&data, count * sizeof(data[0])));
        }
        else {
            // Allocate page-locked host memory
            // - cudaHostAllocDefault: emulates to cudaMallocHost (allocates page-locked memory)
            // - cudaHostAllocPortable: memory considered pinned by all CUDA contexts
            // - cudaHostAllocMapped: allocates a host buffer the device can access directly,
            // generates implicit PCI-e traffic. Likely used by unified memory cudaMallocManaged
            // under the hood (See CUDA C programming guide 6.2.6.3)
            // - cudaHostAllocWriteCombined: bypasses host L1/L2 to improve host-device transfers
            // but results in very slow host-side reads (See CUDA C programming guide 6.2.6.2)
            // unsigned int flags = cudaHostAllocDefault;
            ERRCHK_CUDA_API(cudaAllocHost(&data, count * sizeof(data[0]), flags));
        }
    }

    // Destructor
    ~DeviceBuffer()
    {
        if (on_device) {
            ERRCHK_CUDA_API(cudaFree(data));
        }
        else {
            ERRCHK_CUDA_API(cudaFreeHost(data));
        }
        data      = nullptr;
        stream    = nullptr;
        on_device = false;
        count     = 0;
    }

    // Delete all other types of constructors
    DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
    DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
    DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator

    void migrate(DeviceBuffer<T>& out) const
    {
        ERRCHK(count == out.count);
        cudaMemcpyKind kind;
        if (on_device) {
            if (out.on_device)
                kind = cudaMemcpyDeviceToDevice;
            else
                kind = cudaMemcpyDeviceToHost;
        }
        else {
            if (out.on_device)
                kind = cudaMemcpyHostToDevice;
            else
                kind = cudaMemcpyHostToHost;
        }

        ERRCHK(count == out.count);
        ERRCHK_CUDA_API(cudaMemcpy(out.data, data, count * sizeof(data[0]), kind));
    };
};

template <typename T> struct DeviceToHostBufferExchangeTask {
    std::unique_ptr<DeviceBuffer<T>> host_staging_buffer;
    std::unique_ptr<DeviceBuffer<T>> device_staging_buffer;
    cudaStream_t stream;

    DeviceBufferExchangeTask(const size_t max_count)
        : host_staging_buffer(
              std::make_unique<DeviceBuffer<T>>(max_count, false, cudaAllocHostDefault)),
          device_staging_buffer(std::make_unique<DeviceBuffer<T>>(max_count, true)),
          stream(nullptr)
    {
    }

    ~DeviceBufferExchangeTask() { ERRCHK(stream == nullptr); }

    // Delete all other types of constructors
    DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
    DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
    DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator

    void launch_dtoh(const DeviceBuffer<T>& input)
    {
        ERRCHK(stream = nullptr);
        ERRCHK(device_staging_buffer.count == input.count);
        input.migrate(device_staging_buffer);
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(
            cudaMemcpyAsync(host_staging_buffer.data, device_staging_buffer.data,
                            device_staging_buffer.count * sizeof(device_staging_buffer.data[0]),
                            cudaMemcpyDeviceToHost, stream));
    }
    void wait_dtoh(DeviceBuffer<T>& output)
    {
        ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in progress");
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));
        stream = nullptr;

        host_staging_buffer.migrate(output);
    }
};

template <typename T> struct HostToDeviceBufferExchangeTask {
    std::unique_ptr<DeviceBuffer<T>> host_staging_buffer;
    std::unique_ptr<DeviceBuffer<T>> device_staging_buffer;
    cudaStream_t stream;

    DeviceBufferExchangeTask(const size_t max_count)
        : host_staging_buffer(
              std::make_unique<DeviceBuffer<T>>(max_count, false, cudaAllocHostWriteCombined)),
          device_staging_buffer(std::make_unique<DeviceBuffer<T>>(max_count, true)),
          stream(nullptr)
    {
    }

    ~DeviceBufferExchangeTask() { ERRCHK(stream == nullptr); }

    // Delete all other types of constructors
    DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
    DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
    DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator

    void launch(const DeviceBuffer<T>& input)
    {
        ERRCHK(stream = nullptr);
        ERRCHK(host_staging_buffer.count == input.count);
        input.migrate(host_staging_buffer);
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(
            cudaMemcpyAsync(device_staging_buffer.data, host_staging_buffer.data,
                            host_staging_buffer.count * sizeof(host_staging_buffer.data[0]),
                            cudaMemcpyHostToDevice, stream));
    }
    void wait(DeviceBuffer<T>& output)
    {
        ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in progress");
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));
        stream = nullptr;

        device_staging_buffer.migrate(output);
    }
};

// template <typename T> struct AsyncDeviceBufferExchangeTask {
//     std::unique_ptr<DeviceBuffer<T>> staging_buffer;
//     cudaStream_t stream;

//     DeviceBufferExchange(const size_t count)
//         : staging_buffer(std::make_unique<DeviceBuffer<T>>(count)), stream(nullptr)
//     {
//     }

//     ~DeviceBufferExchange() { ERRCHK(stream == nullptr); }

//     // Delete all other types of constructors
//     DeviceBuffer(const DeviceBuffer&)            = delete; // Copy constructor
//     DeviceBuffer& operator=(const DeviceBuffer&) = delete; // Copy assignment operator
//     DeviceBuffer(DeviceBuffer&&)                 = delete; // Move constructor
//     DeviceBuffer& operator=(DeviceBuffer&&)      = delete; // Move assignment operator

//     void launch(const DeviceBuffer<T>& input, DeviceBuffer<T>& output)
//     {
//         ERRCHK_EXPR_DESC(stream == nullptr, "Previous migrate_launch was still in progress. Call
//         "
//                                             "migrate_wait after migrate_launch to synchronize.");
//         ERRCHK(input.count == output.count);

//         input.migrate(staging_buffer);

//         cudaMemcpyKind kind;
//         if (on_device) {
//             if (output.on_device)
//                 kind = cudaMemcpyDeviceToDevice;
//             else
//                 kind = cudaMemcpyDeviceToHost;
//         }
//         else {
//             if (output.on_device)
//                 kind = cudaMemcpyHostToDevice;
//             else
//                 kind = cudaMemcpyHostToHost;
//         }

//         ERRCHK(input.count == output.count);
//         ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//         ERRCHK_CUDA_API(
//             cudaMemcpyAsync(output.data, input.data, input.count * sizeof(data[0]), kind,
//             stream));
//     };

//     void wait()
//     {
//         ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in
//         progress"); ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
//         ERRCHK_CUDA_API(cudaStreamDestroy(stream));
//         stream = nullptr;
//     }
// };

// namespace DeviceBuffer {
// } // namespace DeviceBuffer
