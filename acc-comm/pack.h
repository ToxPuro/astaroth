#pragma once

#include "array.h"
#include "buffer.h"

#include "math_utils.h"

template <typename T>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const std::vector<ac::buffer<T, ac::mr::host_memory_resource>*>& inputs,
     ac::buffer<T, ac::mr::host_memory_resource>& output)
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

            output[i + j * block_nelems] = (*inputs[j])[in_idx];
        }
    }
}

template <typename T>
void
unpack(const ac::buffer<T, ac::mr::host_memory_resource>& input, const Shape& mm,
       const Shape& block_shape, const Index& block_offset,
       std::vector<ac::buffer<T, ac::mr::host_memory_resource>*>& outputs)
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

            (*outputs[j])[in_idx] = input[i + j * block_nelems];
        }
    }
}

#if defined(DEVICE_ENABLED)

template <typename T>
void pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
          const std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& inputs,
          ac::buffer<T, ac::mr::device_memory_resource>& output);

template <typename T>
void unpack(const ac::buffer<T, ac::mr::device_memory_resource>& input, const Shape& mm,
            const Shape& block_shape, const Index& block_offset,
            std::vector<ac::buffer<T, ac::mr::device_memory_resource>*>& outputs);

#define PACK_DTYPE double
extern template void
pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
                 const std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& inputs,
                 ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& output);

extern template void
unpack<PACK_DTYPE>(const ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& input,
                   const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
                   std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& outputs);
#undef PACK_DTYPE

#define PACK_DTYPE uint64_t
extern template void
pack<PACK_DTYPE>(const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
                 const std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& inputs,
                 ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& output);

extern template void
unpack<PACK_DTYPE>(const ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>& input,
                   const ac::shape& mm, const ac::shape& block_shape, const ac::index& block_offset,
                   std::vector<ac::buffer<PACK_DTYPE, ac::mr::device_memory_resource>*>& outputs);
#undef PACK_DTYPE

#endif

void test_pack(void);
