#pragma once

#include "buffer.h"

#include "math_utils.h"

template <typename T>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const std::vector<ac::mr::host_ptr<T>>& inputs, ac::mr::host_ptr<T>&& output)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const Shape block_coords{to_spatial(i, block_shape)};

            // Input coords
            const Shape in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
void
unpack(const ac::mr::host_ptr<T>& input, const Shape& mm, const Shape& block_shape,
       const Index& block_offset, std::vector<ac::mr::host_ptr<T>>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const Shape block_coords{to_spatial(i, block_shape)};

            // Input coords
            const Shape in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

#if defined(ACM_DEVICE_ENABLED)

template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const std::vector<ac::mr::device_ptr<T>>& inputs, ac::mr::device_ptr<T>&& output);

template <typename T>
void unpack(const ac::mr::device_ptr<T>& input, const Shape& mm, const Shape& block_shape,
            const Index& block_offset, std::vector<ac::mr::device_ptr<T>>& outputs);

#define PACK_DTYPE double
extern template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape,
                                      const Index& block_offset,
                                      const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                                      ac::mr::device_ptr<PACK_DTYPE>&& output);

extern template void unpack<PACK_DTYPE>(const ac::mr::device_ptr<PACK_DTYPE>& input,
                                        const Shape& mm, const Shape& block_shape,
                                        const Index& block_offset,
                                        std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
extern template void pack<PACK_DTYPE>(const Shape& mm, const Shape& block_shape,
                                      const Index& block_offset,
                                      const std::vector<ac::mr::device_ptr<PACK_DTYPE>>& inputs,
                                      ac::mr::device_ptr<PACK_DTYPE>&& output);

extern template void unpack<PACK_DTYPE>(const ac::mr::device_ptr<PACK_DTYPE>& input,
                                        const Shape& mm, const Shape& block_shape,
                                        const Index& block_offset,
                                        std::vector<ac::mr::device_ptr<PACK_DTYPE>>& outputs);
#undef PACK_DTYPE

#endif

void test_pack(void);
