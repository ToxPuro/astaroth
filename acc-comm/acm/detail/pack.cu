#include "pack.h"

#if defined(ACM_DEVICE_ENABLED)

#include "static_array.h"

constexpr size_t MAX_NDIMS       = 4;
constexpr size_t MAX_N_AGGR_BUFS = 8;

namespace device {

template <typename T, size_t N>
static ac::static_array<T, N>
make_static_array(const ac::vector<T>& in)
{
    ac::static_array<T, N> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = in[i];
    return out;
}

template <typename T, size_t N>
static ac::static_array<T, N>
make_static_array(const std::vector<T>& in)
{
    ac::static_array<T, N> out(in.size());
    for (size_t i{0}; i < in.size(); ++i)
        out[i] = in[i];
    return out;
}

static __device__ uint64_t
to_linear(const ac::static_array<uint64_t, MAX_NDIMS>& coords,
          const ac::static_array<uint64_t, MAX_NDIMS>& shape)
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

static __device__ ac::static_array<uint64_t, MAX_NDIMS>
to_spatial(const uint64_t index, const ac::static_array<uint64_t, MAX_NDIMS>& shape)
{
    ac::static_array<uint64_t, MAX_NDIMS> coords(shape.size());
    for (size_t j{0}; j < shape.size(); ++j) {
        uint64_t divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T>
static __device__ T
prod(const ac::static_array<T, MAX_NDIMS>& arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result{1};
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

template <typename T>
__global__ void
kernel_pack(const ac::static_array<uint64_t, MAX_NDIMS> mm,
            const ac::static_array<uint64_t, MAX_NDIMS> block_shape,
            const ac::static_array<uint64_t, MAX_NDIMS> block_offset,
            const ac::static_array<T*, MAX_N_AGGR_BUFS> inputs, T* output)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{device::prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::static_array<uint64_t, MAX_NDIMS> block_coords{
                device::to_spatial(i, block_shape)};

            // Input coords
            const ac::static_array<uint64_t, MAX_NDIMS> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
__global__ void
kernel_unpack(const T* input, const ac::static_array<uint64_t, MAX_NDIMS> mm,
              const ac::static_array<uint64_t, MAX_NDIMS> block_shape,
              const ac::static_array<uint64_t, MAX_NDIMS> block_offset,
              ac::static_array<T*, MAX_N_AGGR_BUFS> outputs)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::static_array<uint64_t, MAX_NDIMS> block_coords{
                device::to_spatial(i, block_shape)};

            // Input coords
            const ac::static_array<uint64_t, MAX_NDIMS> in_coords{block_offset + block_coords};
            const uint64_t in_idx{device::to_linear(in_coords, mm)};

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

} // namespace device

template <typename T>
static std::vector<T*>
unwrap(const std::vector<ac::mr::device_ptr<T>>& buffers)
{
    std::vector<T*> output;
    for (ac::mr::device_ptr<T> ptr : buffers)
        output.push_back(ptr.data());
    return output;
}

template <typename T>
static std::vector<T*>
unwrap(std::vector<ac::mr::device_ptr<T>>& buffers)
{
    std::vector<T*> output;
    for (ac::mr::device_ptr<T> ptr : buffers)
        output.push_back(ptr.data());
    return output;
}

template <typename T>
void
pack(const Shape& in_mm, const Shape& in_block_shape, const Index& in_block_offset,
     const std::vector<ac::mr::device_ptr<T>>& in_inputs, ac::mr::device_ptr<T>&& in_output)
{
    const uint64_t block_nelems{prod(in_block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const auto mm           = device::make_static_array<uint64_t, MAX_NDIMS>(in_mm);
    const auto block_shape  = device::make_static_array<uint64_t, MAX_NDIMS>(in_block_shape);
    const auto block_offset = device::make_static_array<uint64_t, MAX_NDIMS>(in_block_offset);
    const auto inputs       = device::make_static_array<T*, MAX_N_AGGR_BUFS>(unwrap(in_inputs));
    const auto output       = in_output.data();

    device::kernel_pack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset,
                                                                  inputs, output);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T>
void
unpack(const ac::mr::device_ptr<T>& in_input, const Shape& in_mm, const Shape& in_block_shape,
       const Index& in_block_offset, std::vector<ac::mr::device_ptr<T>>& in_outputs)
{
    const uint64_t block_nelems{prod(in_block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const auto input        = in_input.data();
    const auto mm           = device::make_static_array<uint64_t, MAX_NDIMS>(in_mm);
    const auto block_shape  = device::make_static_array<uint64_t, MAX_NDIMS>(in_block_shape);
    const auto block_offset = device::make_static_array<uint64_t, MAX_NDIMS>(in_block_offset);
    const auto outputs      = device::make_static_array<T*, MAX_N_AGGR_BUFS>(unwrap(in_outputs));

    device::kernel_unpack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input, mm, block_shape,
                                                                    block_offset, outputs);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();

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
          const std::vector<ac::mr::device_ptr<T>>& inputs, ac::mr::device_ptr<T>&& output);

template <typename T>
void unpack(const ac::mr::device_ptr<T>& input, const Shape& mm, const Shape& block_shape,
            const Index& block_offset, std::vector<ac::mr::device_ptr<T>>& outputs);

#define PACK_DTYPE double
template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                               const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                               ac::mr::device_ptr<PACK_DTYPE>&& output);

template void unpack<PACK_DTYPE>(const ac::mr::device_ptr<PACK_DTYPE>& input, const Shape& mm,
                                 const Shape& block_shape, const Index& block_offset,
                                 std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                               const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                               ac::mr::device_ptr<PACK_DTYPE>&& output);

template void unpack<PACK_DTYPE>(const ac::mr::device_ptr<PACK_DTYPE>& input, const Shape& mm,
                                 const Shape& block_shape, const Index& block_offset,
                                 std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#endif
