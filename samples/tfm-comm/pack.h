#pragma once
#include "buffer.h"
#include "datatypes.h"

#if defined(CUDA_ENABLED)
#include "errchk_cuda.h"
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include "errchk_cuda.h"
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#include "errchk.h"
#endif

#include "math_utils.h"

template <typename T, size_t N, size_t M>
void
pack(const Shape<N>& mm, const Shape<N>& block_shape, const Index<N>& block_offset,
     const ac::array<T*, M>& inputs, Buffer<T, HostMemoryResource>& output)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {

            // Block coords
            const Index<N> block_coords = to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;

            const uint64_t in_idx = to_linear(in_coords, mm);
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t N, size_t M>
void
unpack(const Buffer<T, HostMemoryResource>& input, const Shape<N>& mm, const Shape<N>& block_shape,
       const Index<N>& block_offset, ac::array<T*, M>& outputs)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < outputs.size(); ++j) {

            // Block coords
            const Index<N> block_coords = to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;

            const uint64_t in_idx = to_linear(in_coords, mm);
            ERRCHK(in_idx < prod(mm));

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

#if defined(DEVICE_ENABLED)
constexpr size_t MAX_PACK_COUNT = 1;

template <typename T, size_t N, size_t M>
void
pack(const Shape<N>& mm, const Shape<N>& block_shape, const Index<N>& block_offset,
     const ac::array<T*, M>& inputs, Buffer<T, DeviceMemoryResource>& output);

template <typename T, size_t N, size_t M>
void
unpack(const Buffer<T, DeviceMemoryResource>& input, const Shape<N>& mm, const Shape<N>& block_shape,
       const Index<N>& block_offset, ac::array<T*, M>& outputs);

extern template
void
pack<AcReal, NDIMS, MAX_PACK_COUNT>(const Shape<NDIMS>& mm, const Shape<NDIMS>& block_shape, const Index<NDIMS>& block_offset,
     const ac::array<AcReal*, MAX_PACK_COUNT>& inputs, Buffer<AcReal, DeviceMemoryResource>& output);

extern template
void
unpack<AcReal, NDIMS, MAX_PACK_COUNT>(const Buffer<AcReal, DeviceMemoryResource>& input, const Shape<NDIMS>& mm, const Shape<NDIMS>& block_shape,
       const Index<NDIMS>& block_offset, ac::array<AcReal*, MAX_PACK_COUNT>& outputs);
#endif

void test_pack(void);
