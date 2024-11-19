#include "pack.h"

namespace device {
template <size_t N>
static __device__ uint64_t
to_linear(const Index<N>& coords, const Shape<N>& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.size(); ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <size_t N>
static __device__ Index<N>
to_spatial(const uint64_t index, const Shape<N>& shape)
{
    Index<N> coords;
    for (size_t j = 0; j < shape.size(); ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
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
    for (size_t i = 0; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_pack(const Shape<N> mm, const Shape<N> block_shape, const Index<N> block_offset,
            const ac::array<T*, M> inputs, T* output)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{device::prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < inputs.size(); ++j) {

            // Block coords
            const Index<N> block_coords = device::to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;
            const uint64_t in_idx    = device::to_linear(in_coords, mm);

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_unpack(const T* input, const Shape<N> mm, const Shape<N> block_shape,
              const Index<N> block_offset, ac::array<T*, M> outputs)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < outputs.size(); ++j) {

            // Block coords
            const Index<N> block_coords = device::to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;
            const uint64_t in_idx    = device::to_linear(in_coords, mm);

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}
} // namespace device

template <typename T, size_t N, size_t M>
void
pack(const Shape<N>& mm, const Shape<N>& block_shape, const Index<N>& block_offset,
     const ac::array<T*, M>& inputs, Buffer<T, DeviceMemoryResource>& output)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    device::kernel_pack<T, N, M><<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset,
                                                                  inputs, output.data());
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T, size_t N, size_t M>
void
unpack(const Buffer<T, DeviceMemoryResource>& input, const Shape<N>& mm, const Shape<N>& block_shape,
       const Index<N>& block_offset, ac::array<T*, M>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    device::kernel_unpack<T, N, M><<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input.data(), mm, block_shape,
                                                                    block_offset, outputs);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

// Specialization
template
void
pack<AcReal, PACK_NDIMS, PACK_MAX_NAGGR_BUFS>(const Shape<PACK_NDIMS>& mm, const Shape<PACK_NDIMS>& block_shape, const Index<PACK_NDIMS>& block_offset,
     const ac::array<AcReal*, PACK_MAX_NAGGR_BUFS>& inputs, Buffer<AcReal, DeviceMemoryResource>& output);

template
void
unpack<AcReal, PACK_NDIMS, PACK_MAX_NAGGR_BUFS>(const Buffer<AcReal, DeviceMemoryResource>& input, const Shape<PACK_NDIMS>& mm, const Shape<PACK_NDIMS>& block_shape,
       const Index<PACK_NDIMS>& block_offset, ac::array<AcReal*, PACK_MAX_NAGGR_BUFS>& outputs);
