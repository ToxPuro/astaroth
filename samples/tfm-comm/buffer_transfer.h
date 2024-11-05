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

template <typename T> class BufferExchangeTask {
  protected:
    Buffer<T> host_staging_buffer;
    Buffer<T> device_staging_buffer;
    cudaStream_t stream;

  public:
    BufferExchangeTask(Buffer<T> in_host_staging_buffer, Buffer<T> in_device_staging_buffer)
        : host_staging_buffer(std::move(in_host_staging_buffer)),
          device_staging_buffer(std::move(in_device_staging_buffer)),
          stream(nullptr)
    {
    }
};

template <typename T> class HostToDeviceBufferExchangeTask : public BufferExchangeTask<T> {
  public:
    HostToDeviceBufferExchangeTask(const size_t max_count)
        : BufferExchangeTask<T>(Buffer<T>(max_count, BUFFER_HOST_PINNED_WRITE_COMBINED),
                                Buffer<T>(max_count, BUFFER_DEVICE))
    {
    }

    void launch(const Buffer<T>& input)
    {
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
        input.migrate(this->host_staging_buffer);
        ERRCHK_CUDA_API(cudaMemcpyAsync(this->device_staging_buffer.data,
                                        this->host_staging_buffer.data,
                                        this->host_staging_buffer.count *
                                            sizeof(this->host_staging_buffer.data[0]),
                                        cudaMemcpyHostToDevice, this->stream));
    }
    void wait(Buffer<T>& output)
    {
        // ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in
        // progress");
        ERRCHK_CUDA_API(cudaStreamSynchronize(this->stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(this->stream));
        this->stream = nullptr;

        this->device_staging_buffer.migrate(output);
    }
};

template <typename T> class DeviceToHostBufferExchangeTask : public BufferExchangeTask<T> {
  public:
    DeviceToHostBufferExchangeTask(const size_t max_count)
        : BufferExchangeTask<T>(Buffer<T>(max_count, BUFFER_HOST_PINNED),
                                Buffer<T>(max_count, BUFFER_DEVICE))
    {
    }

    void launch(const Buffer<T>& input)
    {
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
        input.migrate(this->device_staging_buffer);
        ERRCHK_CUDA_API(cudaMemcpyAsync(this->host_staging_buffer.data,
                                        this->device_staging_buffer.data,
                                        this->device_staging_buffer.count *
                                            sizeof(this->device_staging_buffer.data[0]),
                                        cudaMemcpyDeviceToHost, this->stream));
    }
    void wait(Buffer<T>& output)
    { // ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in
        // progress");
        ERRCHK_CUDA_API(cudaStreamSynchronize(this->stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(this->stream));
        this->stream = nullptr;

        this->host_staging_buffer.migrate(output);
    }
};

// enum BufferExchangeDirection {
//     BUFFER_EXCHANGE_DTOH,
//     BUFFER_EXCHANGE_HTOD,
// };

// template <typename T> class BufferExchangeTask {
//   private:
//     Buffer<T> host_staging_buffer;
//     Buffer<T> device_staging_buffer;
//     cudaStream_t stream;
//     BufferExchangeDirection type;

//   public:
//     BufferExchangeTask(const size_t max_count, const BufferExchangeDirection in_type)
//         : stream(nullptr), type(type)
//     {
//         if (type == BUFFER_EXCHANGE_DTOH) {
//             host_staging_buffer   = Buffer<T>(max_count, BUFFER_HOST_PINNED);
//             device_staging_buffer = Buffer<T>(max_count, BUFFER_DEVICE);
//         }
//         else if (type == BUFFER_EXCHANGE_HTOD) {
//             host_staging_buffer   = Buffer<T>(max_count, BUFFER_HOST_PINNED_WRITE_COMBINED);
//             device_staging_buffer = Buffer<T>(max_count, BUFFER_DEVICE);
//         }
//         else {
//             ERRCHK_EXPR_DESC(false, "Invalid type");
//         }
//     }

//     ~BufferExchangeTask() { WARNCHK(stream == nullptr); }

//     // Delete all other types of constructors
//     BufferExchangeTask(const BufferExchangeTask&)            = delete; // Copy constructor
//     BufferExchangeTask& operator=(const BufferExchangeTask&) = delete; // Copy assignment
//     operator BufferExchangeTask(BufferExchangeTask&&)        = delete; // Move
//     constructor BufferExchangeTask&
//     operator=(BufferExchangeTask&&) = delete; // Move assignment operator

//     void launch(const Buffer<T>& input)
//     {
//         ERRCHK(!stream);
//         ERRCHK(input.count == host_staging_buffer.count);
//         ERRCHK(input.count == device_staging_buffer.count);
// #if defined(DEVICE_ENABLED)
//         ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//         if (type == BUFFER_EXCHANGE_DTOH) {
//             input.migrate(host_staging_buffer);
//             ERRCHK_CUDA_API(
//                 cudaMemcpyAsync(device_staging_buffer.data, host_staging_buffer.data,
//                                 host_staging_buffer.count * sizeof(host_staging_buffer.data[0]),
//                                 cudaMemcpyDeviceToHost, stream));
//         }
//         else if (type == BUFFER_EXCHANGE_HTOD) {
//             input.migrate(host_staging_buffer);
//             ERRCHK_CUDA_API(
//                 cudaMemcpyAsync(device_staging_buffer.data, host_staging_buffer.data,
//                                 host_staging_buffer.count * sizeof(host_staging_buffer.data[0]),
//                                 cudaMemcpyDeviceToHost, stream));
//         }
//         else {
//             ERRCHK(false);
//         }
// #else
//         WARNING_DESC("Device code was not enabled, migrating synchronously within host");
//         stream = &stream;
//         host_staging_buffer.migrate(device_staging_buffer);
// #endif
//     }
//     void wait(Buffer<T>& output)
//     {
//         ERRCHK_EXPR_DESC(stream, "Function called but there was no memory operation in
//         progress");
// #if defined(DEVICE_ENABLED)
//         ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
//         ERRCHK_CUDA_API(cudaStreamDestroy(stream));
// #endif
//         stream = nullptr;

//         device_staging_buffer.migrate(output);
//     }
// };
