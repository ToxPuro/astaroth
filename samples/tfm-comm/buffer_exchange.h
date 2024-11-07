#pragma once

#include <memory>

#include "buffer.h"
#include "cuda_utils.h"

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

template <typename T, typename FirstStageResource, typename SecondStageResource>
class BufferExchangeTask {
  protected:
    Buffer<T, FirstStageResource> first_stage_buffer;
    Buffer<T, SecondStageResource> second_stage_buffer;
    // std::unique_ptr<cudaStream_t, decltype(&cuda_stream_destroy)> stream_ptr;
    CUDAStream stream;
    bool in_progress;

  public:
    BufferExchangeTask(const size_t max_count)
        : first_stage_buffer(max_count),
          second_stage_buffer(max_count),
          //   stream_ptr{cuda_stream_create(), &cuda_stream_destroy},
          stream{},
          in_progress{false}
    {
    }

    template <typename MemoryResource> void launch(const Buffer<T, MemoryResource>& in)
    {
        ERRCHK(!in_progress);
        in_progress = true;

        // Ensure that the input resource and the first-stage buffer is in the same memory space
        if constexpr (std::is_base_of<DeviceMemoryResource, FirstStageResource>::value) {
            static_assert(std::is_base_of<DeviceMemoryResource, MemoryResource>::value);
        }
        else {
            static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        }

        PRINT_LOG("migrating to first-stage buffer");
        migrate(in, first_stage_buffer);
        PRINT_LOG("async migrate to second-stage buffer");
        // migrate_async(*stream_ptr, first_stage_buffer, second_stage_buffer);
        migrate_async(stream.value(), first_stage_buffer, second_stage_buffer);
    }

    template <typename MemoryResource> void wait(Buffer<T, MemoryResource>& out)
    {
        ERRCHK(in_progress);
#if defined(DEVICE_ENABLED)
        // ERRCHK_CUDA_API(cudaStreamSynchronize(*stream_ptr));
        // ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        stream.synchronize();
#endif

        // Ensure that the output resource and the second-stage buffer is in the same memory space
        if constexpr (std::is_base_of<DeviceMemoryResource, SecondStageResource>::value) {
            static_assert(std::is_base_of<DeviceMemoryResource, MemoryResource>::value);
        }
        else {
            static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        }

        migrate(second_stage_buffer, out);
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
