#include "random_experimental.h"

#include "acm/detail/type_conversion.h"

#include "acm/detail/print_debug.h"

namespace acm::experimental {

#if defined(ACM_DEVICE_ENABLED)

__host__ __device__ uint64_t
xorshift(const uint64_t state)
{
    uint64_t x{state};
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

// Workaround: not allowed to call from device code (CUDA only)
constexpr double uint64_t_max_double{static_cast<double>(std::numeric_limits<uint64_t>::max())};

namespace device {

__global__ void
randomize(const uint64_t seed, const size_t count, double* arr)
{
    const size_t i{static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (i >= count)
        return;

    const double x{static_cast<double>(xorshift(seed + xorshift(static_cast<uint64_t>(i))))};
    const double n{uint64_t_max_double};
    arr[i] = x / n;
}

__global__ void
randomize(const uint64_t seed, const size_t count, uint64_t* arr)
{
    const size_t i{static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (i >= count)
        return;

    arr[i] = xorshift(seed + xorshift(static_cast<uint64_t>(i)));
}

} // namespace device

template <typename T>
void
randomize(ac::device_view<T> ptr)
{
    constexpr uint64_t initial_seed{123};
    static uint64_t    seed{initial_seed};

    const size_t tpb{256};
    const size_t bpg{(ptr.size() + tpb - 1) / tpb};

    cudaStream_t stream{nullptr};
    ERRCHK_CUDA_API(cudaStreamCreate(&stream));
    device::randomize<<<as<uint32_t>(bpg), as<uint32_t>(tpb), 0, stream>>>(seed,
                                                                           ptr.size(),
                                                                           ptr.data());
    ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
    ERRCHK_CUDA_API(cudaStreamDestroy(stream));

    seed = xorshift(seed);
    if (seed == 0 || seed == initial_seed) {
        seed = initial_seed;
        PRINT_LOG_WARNING("Random seed sequence wrapped around");
    }
}

void
randomize(ac::device_view<double> ptr)
{
    randomize<double>(ptr);
}

void
randomize(ac::device_view<uint64_t> ptr)
{
    randomize<uint64_t>(ptr);
}
#endif

} // namespace acm::experimental
