#pragma once

#include <memory.h>

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
#endif

#if false
class Stream {
  private:
    std::unique_ptr<cudaStream_t, void (*)(cudaStream_t*)> stream;

  public:
    Stream()
        : stream
    {
        []() {
            PRINT_LOG("new stream");
            cudaStream_t* stream = new cudaStream_t;
            // ERRCHK_CUDA_API(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
            ERRCHK_CUDA_API(cudaStreamCreate(stream));
            return stream;
        }(),
            PRINT_LOG("delete stream");
        WARNCHK_CUDA_API(cudaStreamDestroy(*stream));
        delete stream;
    }
    {
    }
    cudaStream_t value() { return *stream; }
    void wait() { ERRCHK_CUDA_API(cudaStreamSynchronize(*stream)); }
}
#endif

#if false
#if defined(DEVICE_ENABLED)
class Stream {
  private:
    static cudaStream_t* alloc(const unsigned int flags)
    {
        PRINT_LOG("new stream");
        cudaStream_t* stream = new cudaStream_t;
        ERRCHK_CUDA_API(cudaStreamCreateWithFlags(stream, flags));
        // ERRCHK_CUDA_API(cudaStreamCreate(stream));
        return stream;
    }

    static void dealloc(cudaStream_t* ptr)
    {
        PRINT_LOG("del");
        delete ptr;
    }

    std::unique_ptr<cudaStream_t, decltype(&dealloc)> stream;

  public:
    Stream(const unsigned int flags = cudaStreamDefault)
        : stream{alloc(flags), dealloc}
    {
    }

    cudaStream_t value() { return *stream; }
    void wait() {}
};
#else
class Stream {
  private:
    unsigned int flags;

  public:
    Stream(const unsigned int in_flags = 0)
        : flags(in_flags)
    {
    }
    unsigned int value() { return flags; }
    void wait() { /* inop*/ }
};
#endif
#endif
