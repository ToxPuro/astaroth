#include "pack.h"

template <typename T, typename MemoryResource>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const PackPtrArray<T*>& inputs, Buffer<T, MemoryResource>& output)
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

template <typename T, typename MemoryResource>
void
unpack(const Buffer<T, MemoryResource>& input, const Shape& mm, const Shape& block_shape,
       const Index& block_offset, PackPtrArray<T*>& outputs)
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

template void pack<AcReal, HostMemoryResource>(const Shape&, const Shape&, const Index&,
                                               const PackPtrArray<AcReal*>&,
                                               Buffer<AcReal, HostMemoryResource>&);

template void unpack<AcReal, HostMemoryResource>(const Buffer<AcReal, HostMemoryResource>&,
                                                 const Shape&, const Shape&, const Index&,
                                                 PackPtrArray<AcReal*>&);

template void pack<uint64_t, HostMemoryResource>(const Shape&, const Shape&, const Index&,
                                                 const PackPtrArray<uint64_t*>&,
                                                 Buffer<uint64_t, HostMemoryResource>&);

template void unpack<uint64_t, HostMemoryResource>(const Buffer<uint64_t, HostMemoryResource>&,
                                                   const Shape&, const Shape&, const Index&,
                                                   PackPtrArray<uint64_t*>&);

void
test_pack(void)
{
    const size_t count = 10;
    const size_t rr    = 1;
    Buffer<uint64_t, HostMemoryResource> hin(count);
    Buffer<uint64_t, DeviceMemoryResource> din(count);
    Buffer<uint64_t, DeviceMemoryResource> dout(count - 2 * rr);
    Buffer<uint64_t, HostMemoryResource> hout(count - 2 * rr);
    hin.arange();
    hout.fill(0);
    migrate(hin, din);

    Shape mm{count};
    Shape block_shape{count - 2 * rr};
    Index block_offset{rr};
    PackPtrArray<uint64_t*> inputs{din.data()};
    pack(mm, block_shape, block_offset, inputs, dout);
    migrate(dout, hout);
    // hout.display();
    ERRCHK(hout[0] == 1);
    ERRCHK(hout[1] == 2);
    ERRCHK(hout[2] == 3);
    ERRCHK(hout[3] == 4);
    ERRCHK(hout[4] == 5);
    ERRCHK(hout[5] == 6);
    ERRCHK(hout[6] == 7);
    ERRCHK(hout[7] == 8);
}
