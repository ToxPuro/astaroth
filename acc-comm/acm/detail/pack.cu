#include "pack.h"

#if defined(ACM_DEVICE_ENABLED)

#include "type_conversion.h"

constexpr size_t MAX_NDIMS{4};
constexpr size_t MAX_N_AGGR_BUFS{12};

using shape_t                       = ac::static_ntuple<uint64_t, MAX_NDIMS>;
using index_t                       = ac::static_ntuple<uint64_t, MAX_NDIMS>;
template <typename T> using array_t = ac::static_ntuple<T, MAX_N_AGGR_BUFS>;

namespace device {

template <typename T>
__global__ void
pack(const shape_t mm, const shape_t block_shape, const index_t block_offset,
     const array_t<T*> inputs, T* output)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const index_t block_coords{to_spatial(i, block_shape)};

            // Input coords
            const index_t  in_coords{block_offset + block_coords};
            const uint64_t in_idx{to_linear(in_coords, mm)};

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
__global__ void
unpack(const T* input, const shape_t mm, const shape_t block_shape, const index_t block_offset,
       array_t<T*> outputs)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const index_t block_coords{to_spatial(i, block_shape)};

            // Input coords
            const index_t  in_coords{block_offset + block_coords};
            const uint64_t in_idx{to_linear(in_coords, mm)};

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

} // namespace device

template <typename T>
static std::vector<T*>
unwrap(const std::vector<ac::mr::device_pointer<T>>& buffers)
{
    std::vector<T*> output;
    for (ac::mr::device_pointer<T> ptr : buffers)
        output.push_back(ptr.data());
    return output;
}

template <typename T>
void
pack(const Shape& in_mm, const Shape& in_block_shape, const Index& in_block_offset,
     const std::vector<ac::mr::device_pointer<T>>& in_inputs, ac::mr::device_pointer<T> in_output)
{
    ERRCHK_EXPR_DESC(in_mm.size() <= MAX_NDIMS,
                     "Max ndims of pack is %zu (got %zu)\n",
                     MAX_NDIMS,
                     in_mm.shape());
    ERRCHK_EXPR_DESC(in_inputs.size() <= MAX_N_AGGR_BUFS,
                     "Gave %zu inputs but MAX_N_AGGR_BUFS is %zu\n",
                     in_inputs.size(),
                     MAX_N_AGGR_BUFS);

    const uint64_t block_nelems{prod(in_block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const shape_t     mm{in_mm};
    const shape_t     block_shape{in_block_shape};
    const shape_t     block_offset{in_block_offset};
    const array_t<T*> inputs{unwrap(in_inputs)};
    const auto        output{in_output.data()};

    device::pack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm,
                                                           block_shape,
                                                           block_offset,
                                                           inputs,
                                                           output);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T>
void
unpack(const ac::mr::device_pointer<T>& in_input, const Shape& in_mm, const Shape& in_block_shape,
       const Index& in_block_offset, std::vector<ac::mr::device_pointer<T>>& in_outputs)
{
    ERRCHK_EXPR_DESC(in_mm.size() <= MAX_NDIMS,
                     "Max ndims of pack is %zu (got %zu)\n",
                     MAX_NDIMS,
                     in_mm.shape());
    ERRCHK_EXPR_DESC(in_outputs.size() <= MAX_N_AGGR_BUFS,
                     "Gave %zu outputs but MAX_N_AGGR_BUFS is %zu\n",
                     in_outputs.size(),
                     MAX_N_AGGR_BUFS);

    const uint64_t block_nelems{prod(in_block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const auto        input{in_input.data()};
    const shape_t     mm{in_mm};
    const shape_t     block_shape{in_block_shape};
    const shape_t     block_offset{in_block_offset};
    const array_t<T*> outputs{unwrap(in_outputs)};

    device::unpack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input,
                                                             mm,
                                                             block_shape,
                                                             block_offset,
                                                             outputs);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

// Specialization
template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const std::vector<ac::mr::device_pointer<T>>& inputs, ac::mr::device_pointer<T> output);

template <typename T>
void unpack(const ac::mr::device_pointer<T>& input, const Shape& mm, const Shape& block_shape,
            const Index& block_offset, std::vector<ac::mr::device_pointer<T>>& outputs);

#define PACK_DTYPE double
template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                               const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                               ac::mr::device_pointer<PACK_DTYPE>                     output);

template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input, const Shape& mm,
                                 const Shape& block_shape, const Index& block_offset,
                                 std::vector<ac::mr::device_pointer<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape, const Index& block_offset,
                               const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                               ac::mr::device_pointer<PACK_DTYPE>                     output);

template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input, const Shape& mm,
                                 const Shape& block_shape, const Index& block_offset,
                                 std::vector<ac::mr::device_pointer<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#endif
