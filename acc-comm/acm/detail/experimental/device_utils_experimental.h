#pragma once
#include <memory>

namespace ac::device {

#if defined(ACM_DEVICE_ENABLED)
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
    std::unique_ptr<cudaStream_t> m_stream;

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

    sync() { ERRCHK_CUDA_API(cudaStreamSynchronize(*m_stream)); }

    cudaStream_t get() const noexcept { return *m_stream; }
};

#endif

} // namespace ac::device
