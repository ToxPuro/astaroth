#include "pack.h"

#if defined(ACM_DEVICE_ENABLED)

#include "type_conversion.h"

namespace acm {

template <size_t N> using shape_t             = ac::static_ntuple<uint64_t, N>;
template <size_t N> using index_t             = ac::static_ntuple<uint64_t, N>;
template <typename T, size_t N> using array_t = ac::static_ntuple<T, N>;

namespace device {

template <typename T, size_t NDIMS, size_t NINPUTS>
__global__ void
pack(const shape_t<NDIMS> mm, const shape_t<NDIMS> block_shape, const index_t<NDIMS> block_offset,
     const array_t<T*, NINPUTS> unpacked, T* packed, const bool do_pack)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j{0}; j < NINPUTS; ++j) {

            // Block coords
            const auto block_coords{to_spatial(i, block_shape)};

            // Input coords
            const auto in_coords{block_offset + block_coords};
            const auto in_idx{to_linear(in_coords, mm)};

            if (do_pack)
                packed[i + j * block_nelems] = unpacked[j][in_idx];
            else
                unpacked[j][in_idx] = packed[i + j * block_nelems];
        }
    }
}

} // namespace device

template <typename T>
static std::vector<T*>
unwrap_data(const std::vector<ac::mr::device_pointer<T>>& buffers)
{
    std::vector<T*> output;
    for (ac::mr::device_pointer<T> ptr : buffers)
        output.push_back(ptr.data());
    return output;
}

template <typename T, size_t NDIMS, size_t NINPUTS>
void
pack(const ac::shape& in_mm, const ac::shape& in_block_shape, const ac::index& in_block_offset,
     const std::vector<ac::mr::device_pointer<T>>& in_inputs, ac::mr::device_pointer<T> in_output,
     const bool do_pack)
{
    ERRCHK(in_inputs.size() * prod(in_block_shape) <= in_output.size());

    const uint64_t block_nelems{prod(in_block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const shape_t<NDIMS>       mm{in_mm};
    const shape_t<NDIMS>       block_shape{in_block_shape};
    const shape_t<NDIMS>       block_offset{in_block_offset};
    const array_t<T*, NINPUTS> inputs{unwrap_data(in_inputs)};
    const auto                 output{in_output.data()};

    device::pack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm,
                                                           block_shape,
                                                           block_offset,
                                                           inputs,
                                                           output,
                                                           do_pack);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T, size_t NDIMS>
void
pack(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
     const std::vector<ac::mr::device_pointer<T>>& inputs, ac::mr::device_pointer<T> output,
     const bool do_pack)
{
    switch (inputs.size()) {
    case 1:
        return pack<T, NDIMS, 1>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 2:
        return pack<T, NDIMS, 2>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 3:
        return pack<T, NDIMS, 3>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 4:
        return pack<T, NDIMS, 4>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 8:
        return pack<T, NDIMS, 8>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 12:
        return pack<T, NDIMS, 12>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 16:
        return pack<T, NDIMS, 16>(mm, block_shape, block_offset, inputs, output, do_pack);
    default:
        ERROR(false, "Unhandled");
    }
}

template <typename T>
void
pack(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
     const std::vector<ac::mr::device_pointer<T>>& inputs, ac::mr::device_pointer<T> output,
     const bool do_pack)
{
    switch (mm.size()) {
    case 1:
        return pack<T, 1>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 2:
        return pack<T, 2>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 3:
        return pack<T, 3>(mm, block_shape, block_offset, inputs, output, do_pack);
    case 4:
        return pack<T, 4>(mm, block_shape, block_offset, inputs, output, do_pack);
    default:
        ERROR(false, "Unhandled");
    }
}

template <typename T>
void
pack(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
     const std::vector<ac::mr::device_pointer<T>>& inputs, ac::mr::device_pointer<T> output)
{
    pack(mm, block_shape, block_offset, inputs, output, true);
}

template <typename T>
void
unpack(const ac::mr::device_pointer<T>& input, const ac::shape& mm, const ac::shape& block_shape,
       const ac::index& block_offset, std::vector<ac::mr::device_pointer<T>> outputs)
{
    pack(mm, block_shape, block_offset, outputs, input, false);
}

// Specialization
#define PACK_DTYPE double
template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                               const ac::index&                                       block_offset,
                               const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                               ac::mr::device_pointer<PACK_DTYPE>                     output);

template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                 const ac::shape& mm, const ac::shape& block_shape,
                                 const ac::index&                                block_offset,
                                 std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                               const ac::index&                                       block_offset,
                               const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                               ac::mr::device_pointer<PACK_DTYPE>                     output);

template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                 const ac::shape& mm, const ac::shape& block_shape,
                                 const ac::index&                                block_offset,
                                 std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);
#undef PACK_DTYPE

#define PACK_DTYPE int
template void pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape,
                               const ac::index&                                       block_offset,
                               const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                               ac::mr::device_pointer<PACK_DTYPE>                     output);

template void unpack<PACK_DTYPE>(const ac::mr::device_pointer<PACK_DTYPE>& input,
                                 const ac::shape& mm, const ac::shape& block_shape,
                                 const ac::index&                                block_offset,
                                 std::vector<ac::mr::device_pointer<PACK_DTYPE>> outputs);
#undef PACK_DTYPE

} // namespace acm

#endif
