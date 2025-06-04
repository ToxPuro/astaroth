#include "transform.h"

#include "datatypes.h"
#include "type_conversion.h"

constexpr size_t MAX_NDIMS{4};
using shape_t = ac::static_ntuple<uint64_t, MAX_NDIMS>;
using index_t = ac::static_ntuple<uint64_t, MAX_NDIMS>;

namespace device {

template <typename T>
__global__ void
transform(const shape_t dims, const shape_t subdims, const index_t offset, const T* in, T* out)
{
    const uint64_t out_idx{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (out_idx >= prod(subdims))
        return;

    const auto out_coords{to_spatial(out_idx, subdims)};
    const auto in_coords{offset + out_coords};
    const auto in_idx{to_linear(in_coords, dims)};
    out[out_idx] = in[in_idx];
}

} // namespace device

namespace ac {

template <typename T>
void
transform(const ac::shape in_dims, const ac::shape in_subdims, const ac::index in_offset,
          const ac::device_view<T> in, ac::device_view<T> out)
{
    const uint64_t block_nelems{prod(in_subdims)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const shape_t dims{in_dims};
    const shape_t subdims{in_subdims};
    const index_t offset{in_offset};

    device::transform<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(dims,
                                                                subdims,
                                                                offset,
                                                                in.data(),
                                                                out.data());

    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template void transform<double>(const ac::shape in_dims, const ac::shape in_subdims,
                                const ac::index in_offset, const ac::device_view<double> in,
                                ac::device_view<double> out);

template void transform<uint64_t>(const ac::shape in_dims, const ac::shape in_subdims,
                                  const ac::index in_offset, const ac::device_view<uint64_t> in,
                                  ac::device_view<uint64_t> out);

} // namespace ac
