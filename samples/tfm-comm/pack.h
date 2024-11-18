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
     const ac::array<T*, M>& inputs, ac::host_vector<T>& output)
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
unpack(const ac::host_vector<T>& input, const Shape<N>& mm, const Shape<N>& block_shape,
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
namespace detail {
template <size_t N>
static __device__ uint64_t
device_to_linear(const Index<N>& coords, const Shape<N>& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <size_t N>
static __device__ Index<N>
device_to_spatial(const uint64_t index, const Shape<N>& shape)
{
    Index<N> coords(shape.count);
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_pack(const Shape<N> mm, const Shape<N> block_shape, const Index<N> block_offset,
            const ac::array<T*, M> inputs, T* output)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < inputs.count; ++j) {

            // Block coords
            const Index<N> block_coords = detail::device_to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;
            const uint64_t in_idx    = detail::device_to_linear(in_coords, mm);

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t N, size_t M>
__global__ void
kernel_unpack(const T* input, const Shape<N> mm, const Shape<N> block_shape,
              const Index<N> block_offset, ac::array<T*, M> outputs)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < outputs.count; ++j) {

            // Block coords
            const Index<N> block_coords = detail::device_to_spatial(i, block_shape);

            // Input coords
            const Index<N> in_coords = block_offset + block_coords;
            const uint64_t in_idx    = detail::device_to_linear(in_coords, mm);

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}
} // namespace detail

template <typename T, size_t N, size_t M>
void
pack(const Shape<N>& mm, const Shape<N>& block_shape, const Index<N>& block_offset,
     const ac::array<T*, M>& inputs, ac::device_vector<T>& output)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    detail::kernel_pack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset,
                                                                  inputs, output.data());
    ERRCHK_CUDA_KERMEL();
    cudaDeviceSynchronize();
}

template <typename T, size_t N, size_t M>
void
unpack(const ac::device_vector<T>& input, const Shape<N>& mm, const Shape<N>& block_shape,
       const Index<N>& block_offset, ac::array<T*, M>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    detail::kernel_unpack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input.data(), mm, block_shape,
                                                                    block_offset, outputs);
    ERRCHK_CUDA_KERMEL();
    cudaDeviceSynchronize();
}
#endif

void test_pack(void);
