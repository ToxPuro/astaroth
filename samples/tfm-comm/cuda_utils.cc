#include "cuda_utils.h"

cudaStream_t*
cuda_stream_create(const unsigned int flags)
{
    PRINT_LOG("new stream");
    cudaStream_t* stream = new cudaStream_t;
#if defined(DEVICE_ENABLED)
    // ERRCHK_CUDA_API(cudaStreamCreate(stream));
    ERRCHK_CUDA_API(cudaStreamCreateWithFlags(stream, flags));
#else
    *stream = nullptr;
#endif
    return stream;
}

void
cuda_stream_destroy(cudaStream_t* stream) noexcept
{
    PRINT_LOG("delete stream");
#if defined(DEVICE_ENABLED)
    WARNCHK_CUDA_API(cudaStreamDestroy(*stream));
#endif
    delete stream;
}
