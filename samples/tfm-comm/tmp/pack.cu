#include "pack.h"

namespace device {
template <size_t N>
static __device__ uint64_t
to_linear(const ac::shape<N>& coords, const ac::shape<N>& shape)
{
    uint64_t result{0};
    for (size_t j{0}; j < shape.size(); ++j) {
        uint64_t factor{1};
        for (size_t i{0}; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <size_t N>
static __device__ ac::shape<N>
to_spatial(const uint64_t index, const ac::shape<N>& shape)
{
    ac::shape<N> coords;
    for (size_t j{0}; j < shape.size(); ++j) {
        uint64_t divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T, size_t N>
static __device__ T
prod(const ac::array<T, N>& arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result = 1;
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_pack(const ac::shape<N> mm, const ac::shape<N> block_shape, const ac::shape<N> block_offset,
            const ac::array<T*, M> inputs, T* output)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{device::prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::shape<N> block_coords{device::to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<N> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_unpack(const T* input, const ac::shape<N> mm, const ac::shape<N> block_shape,
              const ac::shape<N> block_offset, ac::array<T*, M> outputs)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::shape<N> block_coords{device::to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<N> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}
} // namespace device

template <typename T, size_t N, typename MemoryResource>
void
pack(const ac::shape<N>& mm, const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
     const std::vector<T*>& inputs, T* output)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    constexpr size_t ninputs = 1;
    ERRCHK(inputs.size() == ninputs);
    std::array<T, ninputs> input_array{inputs};
    device::kernel_pack<T, N, M>
        <<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset, input_array,
                                                   output.data());
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T, size_t N, typename MemoryResource>
void
unpack(const T* input, const ac::shape<N>& mm, const ac::shape<N>& block_shape,
       const ac::index<N>& block_offset, std::vector<T*>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    constexpr size_t noutputs = 1;
    ERRCHK(outputs.size() == noutputs);
    std::array<T, noutputs> output_array{outputs};
    device::kernel_unpack<T, N, M>
        <<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input.data(), mm, block_shape, block_offset,
                                                   output_array);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

// Specialization
template void pack<UserType, UserNdims, ac::mr::device_memory_resource>(
    const UserShape& mm, const UserShape& block_shape, const UserIndex& block_offset,
    const std::vector<UserType*>& inputs, UserType* output);

template void unpack<UserType, UserNdims, ac::mr::device_memory_resource>(
    const UserType* input, const UserShape& mm, const UserShape& block_shape,
    const UserShape& block_offset, std::vector<UserType*>& outputs);
