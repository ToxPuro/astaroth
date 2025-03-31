#include "pack.h"

#include "acm/detail/convert.h"
#include "acm/detail/type_conversion.h"

template <size_t N> using shape_t             = ac::static_ntuple<uint64_t, N>;
template <size_t N> using index_t             = ac::static_ntuple<uint64_t, N>;
template <typename T, size_t N> using array_t = ac::static_ntuple<T, N>;

namespace ac::device {

template <typename T, size_t NDIMS, size_t NINPUTS, size_t NSEGMENTS>
__global__ void
pack_batched(const shape_t<NDIMS> mm, const array_t<T*, NINPUTS> unpacked,
             const array_t<shape_t<NDIMS>, NSEGMENTS> dims,
             const array_t<index_t<NDIMS>, NSEGMENTS> offsets, const array_t<T*, NSEGMENTS> packed,
             const bool do_pack)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    const uint64_t curr_input{static_cast<uint64_t>(threadIdx.y) + blockIdx.y * blockDim.y};
    const uint64_t curr_segment{static_cast<uint64_t>(threadIdx.z) + blockIdx.z * blockDim.z};

    if (curr_segment < NSEGMENTS) {
        const auto block_shape{dims[curr_segment]};
        const auto block_nelems{prod(block_shape)};
        if (i < block_nelems && curr_input < NINPUTS) {

            // Block coords
            const auto block_offset{offsets[curr_segment]};
            const auto block_coords{to_spatial(i, block_shape)};

            // Input coords
            const auto in_coords{block_offset + block_coords};
            const auto in_idx{to_linear(in_coords, mm)};

            if (do_pack)
                packed[curr_segment][i + curr_input * block_nelems] = unpacked[curr_input][in_idx];
            else
                unpacked[curr_input][in_idx] = packed[curr_segment][i + curr_input * block_nelems];
        }
    }
}

} // namespace ac::device

namespace ac {

template <typename T, size_t NDIMS, size_t N_UNPACKED, size_t N_SEGMENTS>
void
pack_batched_prototype(const ac::shape&                              in_mm,
                       const std::vector<ac::mr::device_pointer<T>>& in_unpacked,
                       const std::vector<ac::segment>&               in_segments,
                       const std::vector<ac::mr::device_pointer<T>>& in_packed, const bool do_pack)
{
    ERRCHK(same_size(in_segments, in_packed));

    size_t max_block_nelems{0};
    for (const auto& segment : in_segments)
        max_block_nelems = std::max(max_block_nelems, prod(segment.dims));

    const dim3 tpb{256, 1, 1};
    const dim3 bpg{
        as<uint32_t>((max_block_nelems + tpb.x - 1) / tpb.x),
        as<uint32_t>((in_unpacked.size() + tpb.y - 1) / tpb.y),
        as<uint32_t>((in_segments.size() + tpb.z - 1) / tpb.z),
    };

    std::vector<shape_t<NDIMS>> in_dims;
    for (const auto& segment : in_segments)
        in_dims.push_back(shape_t<NDIMS>{segment.dims});

    std::vector<shape_t<NDIMS>> in_offsets;
    for (const auto& segment : in_segments)
        in_offsets.push_back(index_t<NDIMS>{segment.offset});

    const shape_t<NDIMS>          mm{in_mm};
    const array_t<T*, N_UNPACKED> unpacked{unwrap_data(in_unpacked)};
    // const array_t<ac::device::segment<NDIMS>, n_segments> segments{convert<NDIMS>(in_segments};
    const array_t<shape_t<NDIMS>, N_SEGMENTS> dims{in_dims};
    const array_t<index_t<NDIMS>, N_SEGMENTS> offsets{in_offsets};
    const array_t<T*, N_SEGMENTS>             packed{unwrap_data(in_packed)};

    device::pack_batched<<<bpg, tpb>>>(mm, unpacked, dims, offsets, packed, do_pack);
}
} // namespace ac

template <typename T, size_t NDIMS, size_t N_UNPACKED>
void
pack_batched(const ac::shape& mm, const std::vector<ac::mr::device_pointer<T>>& unpacked,
             const std::vector<ac::segment>&        segments,
             std::vector<ac::mr::device_pointer<T>> packed, const bool do_pack)
{
    switch (segments.size()) {
    case 3:
        return ac::pack_batched_prototype<T, NDIMS, N_UNPACKED, 3>(mm,
                                                                   unpacked,
                                                                   segments,
                                                                   packed,
                                                                   do_pack);
    case 9:
        return ac::pack_batched_prototype<T, NDIMS, N_UNPACKED, 9>(mm,
                                                                   unpacked,
                                                                   segments,
                                                                   packed,
                                                                   do_pack);
    case 26:
        return ac::pack_batched_prototype<T, NDIMS, N_UNPACKED, 26>(mm,
                                                                    unpacked,
                                                                    segments,
                                                                    packed,
                                                                    do_pack);
    case 27:
        return ac::pack_batched_prototype<T, NDIMS, N_UNPACKED, 27>(mm,
                                                                    unpacked,
                                                                    segments,
                                                                    packed,
                                                                    do_pack);
    case 81:
        return ac::pack_batched_prototype<T, NDIMS, N_UNPACKED, 81>(mm,
                                                                    unpacked,
                                                                    segments,
                                                                    packed,
                                                                    do_pack);
    default:
        ERROR_DESC("Unhandled %zu", segments.size());
    }
}

template <typename T, size_t NDIMS>
void
pack_batched(const ac::shape& mm, const std::vector<ac::mr::device_pointer<T>>& unpacked,
             const std::vector<ac::segment>&        segments,
             std::vector<ac::mr::device_pointer<T>> packed, const bool do_pack)
{
    switch (unpacked.size()) {
    case 1:
        return pack_batched<T, NDIMS, 1>(mm, unpacked, segments, packed, do_pack);
    case 2:
        return pack_batched<T, NDIMS, 2>(mm, unpacked, segments, packed, do_pack);
    case 3:
        return pack_batched<T, NDIMS, 3>(mm, unpacked, segments, packed, do_pack);
    case 4:
        return pack_batched<T, NDIMS, 4>(mm, unpacked, segments, packed, do_pack);
    default:
        ERROR(false, "Unhandled");
    }
}

template <typename T>
void
pack_batched(const ac::shape& mm, const std::vector<ac::mr::device_pointer<T>>& unpacked,
             const std::vector<ac::segment>&        segments,
             std::vector<ac::mr::device_pointer<T>> packed, const bool do_pack)
{
    switch (mm.size()) {
    case 1:
        return pack_batched<T, 1>(mm, unpacked, segments, packed, do_pack);
    case 2:
        return pack_batched<T, 2>(mm, unpacked, segments, packed, do_pack);
    case 3:
        return pack_batched<T, 3>(mm, unpacked, segments, packed, do_pack);
    case 4:
        return pack_batched<T, 4>(mm, unpacked, segments, packed, do_pack);
    default:
        ERROR(false, "Unhandled");
    }
}

template <typename T>
void
pack_batched(const ac::shape& mm, const std::vector<ac::mr::device_pointer<T>>& unpacked,
             const std::vector<ac::segment>&        segments,
             std::vector<ac::mr::device_pointer<T>> packed)
{
    pack_batched(mm, unpacked, segments, packed, true);
}

template <typename T>
void
unpack_batched(const std::vector<ac::segment>&               segments,
               const std::vector<ac::mr::device_pointer<T>>& packed, const ac::shape& mm,
               std::vector<ac::mr::device_pointer<T>> unpacked)
{
    pack_batched(mm, unpacked, segments, packed, false);
}

// Specialization
#define PACK_DTYPE double
template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE

#define PACK_DTYPE int
template void
pack_batched<PACK_DTYPE>(const ac::shape&                                       mm,
                         const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                         const std::vector<ac::segment>&                        segments,
                         std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);

template void
unpack_batched<PACK_DTYPE>(const std::vector<ac::segment>&                        segments,
                           const std::vector<ac::mr::device_pointer<PACK_DTYPE>>& inputs,
                           const ac::shape&                                       mm,
                           std::vector<ac::mr::device_pointer<PACK_DTYPE>>        outputs);
#undef PACK_DTYPE
