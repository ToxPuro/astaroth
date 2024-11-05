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

#include "buf.h"

template <typename T, typename FirstStageResource, typename SecondStageResource>
class BufferExchangeTask {
  protected:
    GenericBuffer<T, FirstStageResource> first_stage_buffer;
    GenericBuffer<T, SecondStageResource> second_stage_buffer;
    cudaStream_t stream;

  public:
    BufferExchangeTask(const size_t max_count)
        : first_stage_buffer(max_count), second_stage_buffer(max_count), stream(nullptr)
    {
    }

    template <typename MemoryResource> void launch(const GenericBuffer<T, MemoryResource>& in)
    {
        // Ensure that the input resource and the first-stage buffer is in the same memory space
        if constexpr (std::is_base_of<DeviceMemoryResource, FirstStageResource>::value) {
            static_assert(std::is_base_of<DeviceMemoryResource, MemoryResource>::value);
        }
        else {
            static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        }

        ERRCHK(!stream);
        // Nonblocking HIP stream does not synchronize properly for some reason
        // ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ERRCHK_CUDA_API(cudaStreamCreate(&stream));
        migrate(in, first_stage_buffer);
        migrate_async(stream, first_stage_buffer, second_stage_buffer);
    }

    template <typename MemoryResource> void wait(GenericBuffer<T, MemoryResource>& out)
    {
        // Ensure that the output resource and the second-stage buffer is in the same memory space
        if constexpr (std::is_base_of<DeviceMemoryResource, SecondStageResource>::value) {
            static_assert(std::is_base_of<DeviceMemoryResource, MemoryResource>::value);
        }
        else {
            static_assert(std::is_base_of<HostMemoryResource, MemoryResource>::value);
        }

        ERRCHK(stream);
        ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(stream));

        migrate(second_stage_buffer, out);
        stream = nullptr;
    }
};

template <typename T>
using HostToDeviceBufferExchangeTask = BufferExchangeTask<T, PinnedWriteCombinedHostMemoryResource,
                                                          DeviceMemoryResource>;
template <typename T>
using DeviceToHostBufferExchangeTask = BufferExchangeTask<T, DeviceMemoryResource,
                                                          PinnedHostMemoryResource>;
