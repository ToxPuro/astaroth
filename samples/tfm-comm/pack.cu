#include "pack.h"

// static __host__ __device__ uint64_t
// to_linear(const Index& coords, const Shape& shape)
// {
//     uint64_t result = 0;
//     for (size_t j = 0; j < shape.count; ++j) {
//         uint64_t factor = 1;
//         for (size_t i = 0; i < j; ++i)
//             factor *= shape[i];
//         result += coords[j] * factor;
//     }
//     return result;
// }

// static __host__ __device__ Index
// to_spatial(const uint64_t index, const Shape& shape)
// {
//     Index coords(shape.count);
//     for (size_t j = 0; j < shape.count; ++j) {
//         uint64_t divisor = 1;
//         for (size_t i = 0; i < j; ++i)
//             divisor *= shape[i];
//         coords[j] = (index / divisor) % shape[j];
//     }
//     return coords;
// }

template <typename T>
__global__ void
kernel_pack(const Shape mm, const Shape block_shape, const Index block_offset,
            const PackPtrArray<T*> inputs, T* output)
{
}

template <typename T>
__global__ void
kernel_unpack(const T* input, const Shape mm, const Shape block_shape, const Index block_offset,
              PackPtrArray<T*> outputs)
{
}

template <typename T>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const PackPtrArray<T*>& inputs, T* output)
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
unpack(const T* input, const Shape& mm, const Shape& block_shape, const Index& block_offset,
       PackPtrArray<T*>& outputs)
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

template void pack<AcReal>(const Shape&, const Shape&, const Index&, const PackPtrArray<AcReal*>&,
                           AcReal*);

template void unpack<AcReal>(const AcReal*, const Shape&, const Shape&, const Index&,
                             PackPtrArray<AcReal*>&);
