#pragma once

#include <memory>

// #include "buffer.h"
#include "buffer.h"

#if defined(ACM_DEVICE_ENABLED)
#include "errchk_cuda.h"
#else
#include "errchk.h"
using cudaStream_t = unsigned int*;
const unsigned int cudaStreamDefault{0};
#endif

template <typename T, typename FirstStageResource, typename SecondStageResource>
class BufferExchangeTask {
  private:
    ac::buffer<T, FirstStageResource> m_first_stage_buffer;
    ac::buffer<T, SecondStageResource> m_second_stage_buffer;

    cudaStream_t m_stream{nullptr};
    bool m_in_progress{false};

  public:
    explicit BufferExchangeTask(const size_t max_count)
        : m_first_stage_buffer{max_count}, m_second_stage_buffer{max_count}
    {
    }

    BufferExchangeTask(const BufferExchangeTask&)            = delete; // Copy
    BufferExchangeTask& operator=(const BufferExchangeTask&) = delete; // Copy assignment
    BufferExchangeTask(BufferExchangeTask&&) noexcept        = delete; // Move
    BufferExchangeTask& operator=(BufferExchangeTask&&)      = delete; // Move assignment

    ~BufferExchangeTask()
    {
        WARNCHK(!m_in_progress);
        WARNCHK(!m_stream);
    }

    template <typename MemoryResource> void launch(const ac::buffer<T, MemoryResource>& in)
    {
        ERRCHK(!m_in_progress);
        m_in_progress = true;

        // Ensure that the input resource and the first-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<ac::mr::device_memory_resource, FirstStageResource> &&
                       std::is_base_of_v<ac::mr::device_memory_resource, MemoryResource>) ||
                          std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
                      "Input resource must be in the same memory space as the first staging "
                      "buffer");

        PRINT_LOG("migrating to first-stage buffer");
        migrate(in, m_first_stage_buffer);

#if defined(ACM_DEVICE_ENABLED)
        PRINT_LOG("stream create");
        ERRCHK(m_stream == nullptr);
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault));
#endif

        PRINT_LOG("async migrate to second-stage buffer");
        migrate_async(m_stream, m_first_stage_buffer, m_second_stage_buffer);
    }

    template <typename MemoryResource> void wait(ac::buffer<T, MemoryResource>& out)
    {
        ERRCHK(m_in_progress);

        // Ensure that the output resource and the second-stage buffer is in the same memory space
        static_assert((std::is_base_of_v<ac::mr::device_memory_resource, SecondStageResource> &&
                       std::is_base_of_v<ac::mr::device_memory_resource, MemoryResource>) ||
                          std::is_base_of_v<ac::mr::host_memory_resource, MemoryResource>,
                      "Input resource must be in the same memory space as the first staging "
                      "buffer");

// Synchronize m_stream
#if defined(ACM_DEVICE_ENABLED)
        ERRCHK(m_stream != nullptr);
        ERRCHK_CUDA_API(cudaStreamSynchronize(m_stream));
        ERRCHK_CUDA_API(cudaStreamDestroy(m_stream));
        m_stream = nullptr;
#endif

        // Migrate to output buffer
        migrate(m_second_stage_buffer, out);

        // Complete
        m_in_progress = false;
    }
};

template <typename T>
using HostToDeviceBufferExchangeTask = BufferExchangeTask<
    T, ac::mr::pinned_write_combined_host_memory_resource, ac::mr::device_memory_resource>;
template <typename T>
using DeviceToHostBufferExchangeTask = BufferExchangeTask<T, ac::mr::device_memory_resource,
                                                          ac::mr::pinned_host_memory_resource>;

void test_buffer_exchange(void);
