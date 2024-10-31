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

enum BufferExchangeDirection {
    BUFFER_EXCHANGE_DTOH,
    BUFFER_EXCHANGE_HTOD,
};

template <typename T> class BufferExchangeTask {
  private:
    std::unique_ptr<Buffer<T>> second_staging_buffer;
    std::unique_ptr<Buffer<T>> first_staging_buffer;
    cudaStream_t stream;

  public:
    BufferExchangeTask(const size_t max_count, const BufferExchangeDirection type)
        : second_staging_buffer(nullptr), first_staging_buffer(nullptr), stream(nullptr)
    {
        if (type == BUFFER_EXCHANGE_DTOH) {
            first_staging_buffer  = std::make_unique<Buffer<T>>(max_count, BUFFER_DEVICE);
            second_staging_buffer = std::make_unique<Buffer<T>>(max_count, BUFFER_HOST_PINNED);
        }
        else if (type == BUFFER_EXCHANGE_HTOD) {
            first_staging_buffer  = std::make_unique<Buffer<T>>(max_count,
                                                                BUFFER_HOST_PINNED_WRITE_COMBINED);
            second_staging_buffer = std::make_unique<Buffer<T>>(max_count, BUFFER_DEVICE);
        }
        else {
            ERRCHK_EXPR_DESC(false, "Invalid type");
        }
    }

    ~BufferExchangeTask() { WARNCHK(stream == nullptr); }

    // Delete all other types of constructors
    BufferExchangeTask(const BufferExchangeTask&)            = delete; // Copy constructor
    BufferExchangeTask& operator=(const BufferExchangeTask&) = delete; // Copy assignment operator
    BufferExchangeTask(BufferExchangeTask&&)                 = delete; // Move constructor
    BufferExchangeTask& operator=(BufferExchangeTask&&)      = delete; // Move assignment operator

    void launch(const Buffer<T>& input)
    {
        ERRCHK(!stream);
        ERRCHK(input.count == first_staging_buffer->count);
        input.migrate(*first_staging_buffer);
#if defined(DEVICE_ENABLED)
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(
            cudaMemcpyAsync(second_staging_buffer->data, first_staging_buffer->data,
                            first_staging_buffer->count * sizeof(first_staging_buffer->data[0]),
                            cudaMemcpyDeviceToHost, stream));
#else
        WARNING_DESC("Device code was not enabled, migrating synchronously within host");
        stream = &stream;
        first_staging_buffer->migrate(*second_staging_buffer);
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

        second_staging_buffer->migrate(output);
    }
};
