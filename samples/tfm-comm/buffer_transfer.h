#pragma once

#include <memory>

#include "buffer.h"

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
#define cudaStream_t void*
#endif

template <typename T> class DeviceToHostBufferExchangeTask {
  private:
    std::unique_ptr<Buffer<T>> host_staging_buffer;
    std::unique_ptr<Buffer<T>> device_staging_buffer;
    cudaStream_t stream;

  public:
    DeviceToHostBufferExchangeTask(const size_t max_count)
        : host_staging_buffer(std::make_unique<Buffer<T>>(max_count, BUFFER_HOST_PINNED)),
          device_staging_buffer(std::make_unique<Buffer<T>>(max_count, BUFFER_DEVICE)),
          stream(nullptr)
    {
    }

    ~DeviceToHostBufferExchangeTask() { WARNCHK(stream == nullptr); }

    // Delete all other types of constructors
    DeviceToHostBufferExchangeTask(const DeviceToHostBufferExchangeTask&) =
        delete; // Copy constructor
    DeviceToHostBufferExchangeTask&
    operator=(const DeviceToHostBufferExchangeTask&) = delete; // Copy assignment operator
    DeviceToHostBufferExchangeTask(DeviceToHostBufferExchangeTask&&) = delete; // Move constructor
    DeviceToHostBufferExchangeTask&
    operator=(DeviceToHostBufferExchangeTask&&) = delete; // Move assignment operator

    void launch(const Buffer<T>& input)
    {
        ERRCHK(!stream);
        ERRCHK(input.count == device_staging_buffer->count);
        input.migrate(*device_staging_buffer);
#if defined(DEVICE_ENABLED)
        ERRCHK(input.type == BUFFER_DEVICE);
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(
            cudaMemcpyAsync(host_staging_buffer->data, device_staging_buffer->data,
                            device_staging_buffer->count * sizeof(device_staging_buffer->data[0]),
                            cudaMemcpyDeviceToHost, stream));
#else
        WARNING_DESC("Device code was not enabled, migrating synchronously within host");
        stream = &stream;
        device_staging_buffer->migrate(*host_staging_buffer);
#endif
    }
    void wait(Buffer<T>& output)
    {
        ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in progress");
#if defined(DEVICE_ENABLED)
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));
#endif
        stream = nullptr;

        host_staging_buffer->migrate(output);
    }
};

template <typename T> class HostToDeviceBufferExchangeTask {
  private:
    std::unique_ptr<Buffer<T>> host_staging_buffer;
    std::unique_ptr<Buffer<T>> device_staging_buffer;
    cudaStream_t stream;

  public:
    HostToDeviceBufferExchangeTask(const size_t max_count)
        : host_staging_buffer(
              std::make_unique<Buffer<T>>(max_count, BUFFER_HOST_PINNED_WRITE_COMBINED)),
          device_staging_buffer(std::make_unique<Buffer<T>>(max_count, BUFFER_DEVICE)),
          stream(nullptr)
    {
    }

    ~HostToDeviceBufferExchangeTask() { WARNCHK(stream == nullptr); }

    // Delete all other types of constructors
    HostToDeviceBufferExchangeTask(const HostToDeviceBufferExchangeTask&) =
        delete; // Copy constructor
    HostToDeviceBufferExchangeTask&
    operator=(const HostToDeviceBufferExchangeTask&) = delete; // Copy assignment operator
    HostToDeviceBufferExchangeTask(HostToDeviceBufferExchangeTask&&) = delete; // Move constructor
    HostToDeviceBufferExchangeTask&
    operator=(HostToDeviceBufferExchangeTask&&) = delete; // Move assignment operator

    void launch(const Buffer<T>& input)
    {
        ERRCHK(!stream);
        ERRCHK(input.count == host_staging_buffer->count);
        ERRCHK(input.type == BUFFER_HOST ||
               input.type == BUFFER_HOST_PINNED); // Write-combined slow, not allowed
        input.migrate(*host_staging_buffer);
#if defined(DEVICE_ENABLED)
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(
            cudaMemcpyAsync(device_staging_buffer->data, host_staging_buffer->data,
                            host_staging_buffer->count * sizeof(host_staging_buffer->data[0]),
                            cudaMemcpyHostToDevice, stream));
#else
        WARNING_DESC("Device code was not enabled, migrating synchronously within host");
        stream = &stream;
        host_staging_buffer->migrate(*device_staging_buffer);
#endif
    }
    void wait(Buffer<T>& output)
    {
        ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in progress");
#if defined(DEVICE_ENABLED)
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));
#endif
        stream = nullptr;

        device_staging_buffer->migrate(output);
    }
};
