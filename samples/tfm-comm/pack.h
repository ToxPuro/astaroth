#pragma once
#include "datatypes.h"

#include "math_utils.h"

constexpr size_t PACK_MAX_INPUTS         = 27;
template <typename T> using PackPtrArray = StaticArray<T, PACK_MAX_INPUTS>;

template <typename T>
void
pack(const Shape mm, const Shape block_shape, const Index block_offset,
     const PackPtrArray<T> inputs, AcReal* output)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < inputs.count; ++j) {

            // Block coords
            const Index block_coords = to_spatial(i, block_shape);

            // Input coords
            const Index in_coords = block_offset + block_coords;

            const uint64_t in_idx = to_linear(in_coords, mm);
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
void
unpack(const AcReal* input, const Shape mm, const Shape block_shape, const Index block_offset,
       PackPtrArray<T> outputs)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < outputs.count; ++j) {

            // Block coords
            const Index block_coords = to_spatial(i, block_shape);

            // Input coords
            const Index in_coords = block_offset + block_coords;

            const uint64_t in_idx = to_linear(in_coords, mm);
            ERRCHK(in_idx < prod(mm));

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}
