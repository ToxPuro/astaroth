#pragma once

#include "array.h"
#include "buffer.h"

#include "math_utils.h"

template <typename T, size_t N>
void
pack(const ac::shape<N>& mm, const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
     const std::vector<ac::buffer<T, HostMemoryResource>*>& inputs,
     ac::buffer<T, HostMemoryResource>& output)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::shape<N> block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<N> in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = (*inputs[j])[in_idx];
        }
    }
}

template <typename T, size_t N>
void
unpack(const ac::buffer<T, HostMemoryResource>& input, const ac::shape<N>& mm,
       const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
       std::vector<ac::buffer<T, HostMemoryResource>*>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::shape<N> block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<N> in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            (*outputs[j])[in_idx] = input[i + j * block_nelems];
        }
    }
}

#if defined(DEVICE_ENABLED)

template <typename T, size_t N>
void pack(const ac::shape<N>& mm, const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
          const std::vector<ac::buffer<T, DeviceMemoryResource>*>& inputs,
          ac::buffer<T, DeviceMemoryResource>& output);

template <typename T, size_t N>
void unpack(const ac::buffer<T, DeviceMemoryResource>& input, const ac::shape<N>& mm,
            const ac::shape<N>& block_shape, const ac::index<N>& block_offset,
            std::vector<ac::buffer<T, DeviceMemoryResource>*>& outputs);

#define PACK_DTYPE double
#define PACK_NDIMS (1)
extern template void pack<PACK_DTYPE, PACK_NDIMS>(
    const ac::shape<PACK_NDIMS>& mm, const ac::shape<PACK_NDIMS>& block_shape,
    const ac::index<PACK_NDIMS>& block_offset,
    const std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& inputs,
    ac::buffer<PACK_DTYPE, DeviceMemoryResource>& output);

extern template void unpack<PACK_DTYPE, PACK_NDIMS>(
    const ac::buffer<PACK_DTYPE, DeviceMemoryResource>& input, const ac::shape<PACK_NDIMS>& mm,
    const ac::shape<PACK_NDIMS>& block_shape, const ac::index<PACK_NDIMS>& block_offset,
    std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& outputs);
#undef PACK_DTYPE
#undef PACK_NDIMS

#define PACK_DTYPE double
#define PACK_NDIMS (2)
extern template void pack<PACK_DTYPE, PACK_NDIMS>(
    const ac::shape<PACK_NDIMS>& mm, const ac::shape<PACK_NDIMS>& block_shape,
    const ac::index<PACK_NDIMS>& block_offset,
    const std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& inputs,
    ac::buffer<PACK_DTYPE, DeviceMemoryResource>& output);

extern template void unpack<PACK_DTYPE, PACK_NDIMS>(
    const ac::buffer<PACK_DTYPE, DeviceMemoryResource>& input, const ac::shape<PACK_NDIMS>& mm,
    const ac::shape<PACK_NDIMS>& block_shape, const ac::index<PACK_NDIMS>& block_offset,
    std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& outputs);
#undef PACK_DTYPE
#undef PACK_NDIMS

#define PACK_DTYPE double
#define PACK_NDIMS (3)
extern template void pack<PACK_DTYPE, PACK_NDIMS>(
    const ac::shape<PACK_NDIMS>& mm, const ac::shape<PACK_NDIMS>& block_shape,
    const ac::index<PACK_NDIMS>& block_offset,
    const std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& inputs,
    ac::buffer<PACK_DTYPE, DeviceMemoryResource>& output);

extern template void unpack<PACK_DTYPE, PACK_NDIMS>(
    const ac::buffer<PACK_DTYPE, DeviceMemoryResource>& input, const ac::shape<PACK_NDIMS>& mm,
    const ac::shape<PACK_NDIMS>& block_shape, const ac::index<PACK_NDIMS>& block_offset,
    std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& outputs);
#undef PACK_DTYPE
#undef PACK_NDIMS

#define PACK_DTYPE uint64_t
#define PACK_NDIMS (1)
extern template void pack<PACK_DTYPE, PACK_NDIMS>(
    const ac::shape<PACK_NDIMS>& mm, const ac::shape<PACK_NDIMS>& block_shape,
    const ac::index<PACK_NDIMS>& block_offset,
    const std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& inputs,
    ac::buffer<PACK_DTYPE, DeviceMemoryResource>& output);

extern template void unpack<PACK_DTYPE, PACK_NDIMS>(
    const ac::buffer<PACK_DTYPE, DeviceMemoryResource>& input, const ac::shape<PACK_NDIMS>& mm,
    const ac::shape<PACK_NDIMS>& block_shape, const ac::index<PACK_NDIMS>& block_offset,
    std::vector<ac::buffer<PACK_DTYPE, DeviceMemoryResource>*>& outputs);
#undef PACK_DTYPE
#undef PACK_NDIMS

#endif

void test_pack(void);
