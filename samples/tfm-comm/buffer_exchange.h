#pragma once

#include <memory>

#include "buffer.h"

#if defined(CUDA_ENABLED)
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
using cudaStream_t                   = unsigned int*;
const unsigned int cudaStreamDefault{0};
#endif

template <typename T, typename FirstStageResource, typename SecondStageResource>
class BufferExchangeTask {
  private:
    Buffer<T, FirstStageResource> first_stage_buffer;
    Buffer<T, SecondStageResource> second_stage_buffer;

    cudaStream_t stream{nullptr};
    bool in_progress{false};

  public:
    explicit BufferExchangeTask(const size_t max_count)
        : first_stage_buffer{max_count}, second_stage_buffer{max_count}
    {
    }

    BufferExchangeTask(const BufferExchangeTask&)            = delete; // Copy
    BufferExchangeTask& operator=(const BufferExchangeTask&) = delete; // Copy assignment
    BufferExchangeTask(BufferExchangeTask&&) noexcept        = delete; // Move
    BufferExchangeTask& operator=(BufferExchangeTask&&)      = delete; // Move assignment

    ~BufferExchangeTask()
    {
        WARNCHK(!in_progress);
        WARNCHK(!stream);
    }

    template <typename MemoryResource> void launch(const Buffer<T, MemoryResource>& in)
    {
        ERRCHK(!in_progress);
        in_progress = true;

        // Ensure that the input resource and the first-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<DeviceMemoryResource, FirstStageResource> &&
                       std::is_base_of_v<DeviceMemoryResource, MemoryResource>) ||
                          std::is_base_of_v<HostMemoryResource, MemoryResource>,
                      "Input resource must be in the same memory space as the first staging "
                      "buffer");

        PRINT_LOG("migrating to first-stage buffer");
        migrate(in, first_stage_buffer);

#if defined(DEVICE_ENABLED)
        PRINT_LOG("stream create");
        ERRCHK(stream == nullptr);
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
#endif

        PRINT_LOG("async migrate to second-stage buffer");
        migrate_async(stream, first_stage_buffer, second_stage_buffer);
    }

    template <typename MemoryResource> void wait(Buffer<T, MemoryResource>& out)
    {
        ERRCHK(in_progress);

        // Ensure that the output resource and the second-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<DeviceMemoryResource, SecondStageResource> &&
                       std::is_base_of_v<DeviceMemoryResource, MemoryResource>) ||
                          std::is_base_of_v<HostMemoryResource, MemoryResource>,
                      "Input resource must be in the same memory space as the first staging "
                      "buffer");

// Synchronize stream
#if defined(DEVICE_ENABLED)
        ERRCHK(stream != nullptr);
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));
        stream = nullptr;
#endif

        // Migrate to output buffer
        migrate(second_stage_buffer, out);

        // Complete
        in_progress = false;
    }
};

template <typename T>
using HostToDeviceBufferExchangeTask = BufferExchangeTask<T, PinnedWriteCombinedHostMemoryResource,
                                                          DeviceMemoryResource>;
template <typename T>
using DeviceToHostBufferExchangeTask = BufferExchangeTask<T, DeviceMemoryResource,
                                                          PinnedHostMemoryResource>;

void test_buffer_exchange(void);
