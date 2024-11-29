#pragma once

#include <memory>

// #include "buffer.h"
#include "buffer.h"

#if defined(DEVICE_ENABLED)
#include "errchk_cuda.h"
#else
#include "errchk.h"
using cudaStream_t = unsigned int*;
const unsigned int cudaStreamDefault{0};
#endif

template <typename T, typename FirstStageResource, typename SecondStageResource>
class BufferExchangeTask {
  private:
    ac::buffer<T, FirstStageResource> first_stage_buffer;
    ac::buffer<T, SecondStageResource> second_stage_buffer;

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

    template <typename MemoryResource> void launch(const ac::buffer<T, MemoryResource>& in)
    {
        ERRCHK(!in_progress);
        in_progress = true;

        // Ensure that the input resource and the first-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<ac::mr::device_memory_resource, FirstStageResource> &&
                       std::is_base_of_v<ac::mr::device_memory_resource, MemoryResource>) ||
                          std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
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

    template <typename MemoryResource> void wait(ac::buffer<T, MemoryResource>& out)
    {
        ERRCHK(in_progress);

        // Ensure that the output resource and the second-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<ac::mr::device_memory_resource, SecondStageResource> &&
                       std::is_base_of_v<ac::mr::device_memory_resource, MemoryResource>) ||
                          std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
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
using HostToDeviceBufferExchangeTask = BufferExchangeTask<
    T, ac::mr::pinned_write_combined_host_memory_resource, ac::mr::device_memory_resource>;
template <typename T>
using DeviceToHostBufferExchangeTask = BufferExchangeTask<T, ac::mr::device_memory_resource,
                                                          ac::mr::pinned_host_memory_resource>;

void test_buffer_exchange(void);
