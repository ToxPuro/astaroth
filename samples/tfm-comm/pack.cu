#include "pack.h"

static __device__ uint64_t
device_to_linear(const Index& coords, const Shape& shape)
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

static __device__ Index
device_to_spatial(const uint64_t index, const Shape& shape)
{
    Index coords(shape.count);
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T>
__global__ void
kernel_pack(const Shape mm, const Shape block_shape, const Index block_offset,
            const PackPtrArray<T*> inputs, T* output)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < inputs.count; ++j) {

            // Block coords
            const Index block_coords = device_to_spatial(i, block_shape);

            // Input coords
            const Index in_coords = block_offset + block_coords;
            const uint64_t in_idx = device_to_linear(in_coords, mm);

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T>
__global__ void
kernel_unpack(const T* input, const Shape mm, const Shape block_shape, const Index block_offset,
              PackPtrArray<T*> outputs)
{
    const uint64_t i = static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const uint64_t block_nelems{prod(block_shape)};
    if (i < block_nelems) {
        for (size_t j = 0; j < outputs.count; ++j) {

            // Block coords
            const Index block_coords = device_to_spatial(i, block_shape);

            // Input coords
            const Index in_coords = block_offset + block_coords;
            const uint64_t in_idx = device_to_linear(in_coords, mm);

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

template <typename T, typename MemoryResource>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const PackPtrArray<T*>& inputs, Buffer<T, MemoryResource>& output)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    kernel_pack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(mm, block_shape, block_offset, inputs,
                                                          output.data());
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template <typename T, typename MemoryResource>
void
unpack(const Buffer<T, MemoryResource>& input, const Shape& mm, const Shape& block_shape,
       const Index& block_offset, PackPtrArray<T*>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    const uint64_t tpb{256};
    const uint64_t bpg{(block_nelems + tpb - 1) / tpb};
    kernel_unpack<<<as<uint32_t>(bpg), as<uint32_t>(tpb)>>>(input.data(), mm, block_shape,
                                                            block_offset, outputs);
    ERRCHK_CUDA_KERNEL();
    cudaDeviceSynchronize();
}

template void pack<AcReal, DeviceMemoryResource>(const Shape&, const Shape&, const Index&,
                                                 const PackPtrArray<AcReal*>&,
                                                 Buffer<AcReal, DeviceMemoryResource>&);

template void unpack<AcReal, DeviceMemoryResource>(const Buffer<AcReal, DeviceMemoryResource>&,
                                                   const Shape&, const Shape&, const Index&,
                                                   PackPtrArray<AcReal*>&);
