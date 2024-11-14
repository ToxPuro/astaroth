#include "devicetest.h"

// #if defined(HIP_ENABLED)
// #include "hip.h"
// #include <hip/hip_runtime.h>
// #endif

#include "errchk_cuda.h"
#include "type_conversion.h"

template <typename T>
__global__ void
kernel(const size_t count, const T* in, T* out)
{
    const size_t i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    if (i < count)
        out[i] = 2 * in[i];
}

template <typename T>
void
call_device(const cudaStream_t stream, const size_t count, const T* in, T* out)
{
    const size_t tpb{256};
    const size_t bpg{(count + tpb - 1) / count};
    kernel<<<as<uint32_t>(bpg), as<uint32_t>(tpb), 0, stream>>>(count, in, out);
    ERRCHK_CUDA_KERNEL();
}

template void call_device<double>(const cudaStream_t, const size_t, const double*, double*);
