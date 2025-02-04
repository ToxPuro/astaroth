#include "transform.h"

#include "datatypes.h"
#include "type_conversion.h"

constexpr size_t MAX_NDIMS{4};

using StaticShape = ac::static_ntuple<uint64_t, MAX_NDIMS>;
using StaticIndex = ac::static_ntuple<uint64_t, MAX_NDIMS>;

namespace device {

__global__ void
transform(const StaticShape dims, const StaticShape subdims, const StaticIndex offset,
          const DevicePointer in, DevicePointer out)
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
void
transform(const Shape in_dims, const Shape in_subdims, const Index in_offset,
          const DevicePointer in, DevicePointer out)
{
    const uint64_t block_nelems{prod(in_subdims)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};

    const StaticShape dims{in_dims};
    const StaticShape subdims{in_subdims};
    const StaticIndex offset{in_offset};

    device::transform<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(dims, subdims, offset, in, out);

    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}
} // namespace ac
