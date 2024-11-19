#include "devicetest.h"

// #if defined(HIP_ENABLED)
// #include "hip.h"
// #include <hip/hip_runtime.h>
// #endif

#include <iostream>

#include "errchk_cuda.h"
#include "type_conversion.h"

// #include <hip/std/array>

template <typename T>
__global__ void
kernel(const size_t count, const T* in, T* out)
{
    const size_t i{static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (i < count)
        out[i] = 2 * in[i];
}

template <typename T, size_t N>
std::array<T, N> __device__
operator+(const std::array<T, N>& a, const std::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    std::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T>
__global__ void
other_kernel(const std::array<int, 3> values, const size_t count, const T* in, T* out)
{
    const size_t i{static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (i < count) {
        const auto more_values{values + values};
        out[i] = values[0] + values[1] + values[2] + more_values[0];
    }
}

template <typename T>
void
call_device(const cudaStream_t stream, const size_t count, const T* in, T* out)
{
    const size_t tpb{256};
    const size_t bpg{(count + tpb - 1) / count};
    kernel<<<as<uint32_t>(bpg), as<uint32_t>(tpb), 0, stream>>>(count, in, out);
    ERRCHK_CUDA_KERNEL();

    std::cout << "hello from call_device" << std::endl;
    std::array<int, 3> vals{1, 2, 3};
    other_kernel<<<as<uint32_t>(bpg), as<uint32_t>(tpb), 0, stream>>>(vals, count, in, out);
    ERRCHK_CUDA_KERNEL();
}

template void call_device<double>(const cudaStream_t, const size_t, const double*, double*);
