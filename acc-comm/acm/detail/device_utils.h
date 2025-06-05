#pragma once

#if defined(ACM_DEVICE_ENABLED)

#include <memory>

#include "acm/detail/allocator.h"
#include "acm/detail/cuda_utils.h" // TODO merge with device_utils.h
#include "acm/detail/errchk_cuda.h"

namespace ac::device {
// class event {
//   private:
//     std::unique_ptr<cudaEvent_t> m_event;

//   public:
//     /**
//      * Wraps a CUDA event.
//      */
//     event()
//         : m_event{[]() {
//                       auto ptr{new cudaEvent_t{nullptr}};
//                       ERRCHK_CUDA_API(cudaEventCreate(ptr));
//                       return ptr;
//                   }(),
//                   [](cudaEvent_t* ptr) {
//                       ERRCHK(*ptr != nullptr);
//                       ERRCHK_CUDA_API(cudaEventDestroy(*ptr));
//                       delete ptr;
//                   }}
//     {
//     }
// };

class stream {
  private:
    std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> m_stream;

  public:
    /**
     * Wraps a CUDA stream.
     */
    stream()
        : m_stream{[]() {
                       auto ptr{new cudaStream_t{nullptr}};
                       ERRCHK_CUDA_API(cudaStreamCreate(ptr));
                       return ptr;
                   }(),
                   [](cudaStream_t* ptr) {
                       ERRCHK(*ptr != nullptr);
                       ERRCHK(cudaStreamDestroy(*ptr) == cudaSuccess);
                       delete ptr;
                   }}
    {
    }

    void wait() { ERRCHK_CUDA_API(cudaStreamSynchronize(*m_stream)); }

    cudaStream_t get() const noexcept { return *m_stream; }
};

template <typename InAllocator, typename OutAllocator>
constexpr cudaMemcpyKind
get_kind()
{
    if constexpr (std::is_base_of_v<ac::mr::device_allocator, InAllocator>) {
        if constexpr (std::is_base_of_v<ac::mr::device_allocator, OutAllocator>) {
            PRINT_LOG_TRACE("dtod");
            return cudaMemcpyDeviceToDevice;
        }
        else {
            PRINT_LOG_TRACE("dtoh");
            return cudaMemcpyDeviceToHost;
        }
    }
    else {
        if constexpr (std::is_base_of_v<ac::mr::device_allocator, OutAllocator>) {
            PRINT_LOG_TRACE("htod");
            return cudaMemcpyHostToDevice;
        }
        else {
            PRINT_LOG_TRACE("htoh");
            return cudaMemcpyHostToHost;
        }
    }
}

} // namespace ac::device

#endif
