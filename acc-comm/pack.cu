#include "pack.h"

#if defined(DEVICE_ENABLED)

#if 0
namespace device {
template <size_t N>
static __device__ uint64_t
to_linear(const ac::array<uint64_t, N>& coords, const ac::array<uint64_t, N>& shape)
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
static __device__ ac::array<uint64_t, N>
to_spatial(const uint64_t index, const ac::array<uint64_t, N>& shape)
{
    ac::array<uint64_t, N> coords;
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
prod(const ac::vector<T>& arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result = 1;
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_pack(const ac::array<uint64_t, N> mm, const ac::array<uint64_t, N> block_shape,
            const ac::array<uint64_t, N> block_offset, const ac::array<T*, M> inputs, T* output)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{device::prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::array<uint64_t, N> block_coords{device::to_spatial(i, block_shape)};

            // Input coords
            const ac::array<uint64_t, N> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_unpack(const T* input, const ac::array<uint64_t, N> mm,
              const ac::array<uint64_t, N> block_shape, const ac::array<uint64_t, N> block_offset,
              ac::array<T*, M> outputs)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::array<uint64_t, N> block_coords{device::to_spatial(i, block_shape)};

            // Input coords
            const ac::array<uint64_t, N> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}
} // namespace device
#endif

template <typename T>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& inputs,
     ac::buffer<T, ac::mr::device_memory_resource>& output)
{
    // const uint64_t block_nelems{prod(block_shape)};
    // const uint64_t tpb{256};
    // const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    // switch (inputs.size()) {
    // case 1: {
    //     ac::vector<T*> input_array{inputs[0]->data()};
    //     device::kernel_pack<T, N, 1>
    //         <<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset,
    //         input_array,
    //                                                    output.data());
    //     break;
    // }
    // default:
    //     ERRCHK_EXPR_DESC(false, "input size %zu not supported", inputs.size());
    // }
    // ERRCHK_CUDA_KERNEL();
    // cudaDeviceSynchronize();
}

template <typename T>
void
unpack(const ac::buffer<T, ac::mr::device_memory_resource>& input, const Shape& mm,
       const Shape& block_shape, const Index& block_offset,
       std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& outputs)
{
    // const uint64_t block_nelems{prod(block_shape)};
    // const uint64_t tpb{256};
    // const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    // switch (outputs.size()) {
    // case 1: {
    //     ac::vector<T*> output_array{outputs[0]->data()};
    //     device::kernel_unpack<T, N, 1>
    //         <<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input.data(), mm, block_shape,
    //         block_offset,
    //                                                    output_array);
    //     break;
    // }
    // default:
    //     ERRCHK_EXPR_DESC(false, "input size %zu not supported", inputs.size());
    // }
    // ERRCHK_CUDA_KERNEL();
    // cudaDeviceSynchronize();
}

// Specialization
template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& inputs,
          ac::buffer<T, ac::mr::device_memory_resource>& output);

template <typename T>
void unpack(const ac::buffer<T, ac::mr::device_memory_resource>& input, const Shape& mm,
            const Shape& block_shape, const Index& block_offset,
            std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& outputs);

template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& inputs,
          ac::buffer<T, ac::mr::device_memory_resource>& output);

template <typename T>
void unpack(const ac::buffer<T, ac::mr::device_memory_resource>& input, const Shape& mm,
            const Shape& block_shape, const Index& block_offset,
            std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& outputs);

#define PACK_DTYPE double
template void
pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                 const std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& inputs,
                 ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& output);

template void
unpack<PACK_DTYPE>(const ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& input,
                   const Shape& mm, const Shape& block_shape, const Index& block_offset,
                   std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
template void
pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                 const std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& inputs,
                 ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& output);

template void
unpack<PACK_DTYPE>(const ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& input,
                   const Shape& mm, const Shape& block_shape, const Index& block_offset,
                   std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& outputs);
#undef PACK_DTYPE

#endif
