#include "pack.h"

#include <numeric>

template <typename T, size_t ndims>
void
pack(const ac::shape<ndims>& mm, const ac::shape<ndims>& block_shape,
     const ac::shape<ndims>& block_offset, const std::vector<T*>& inputs, T* output)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < inputs.size(); ++j) {

            // Block coords
            const ac::shape<ndims> block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<ndims> in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            output[i + j * block_nelems] = inputs[j][in_idx];
        }
    }
}

template <typename T, size_t ndims>
void
unpack(const T* input, const ac::shape<ndims>& mm, const ac::shape<ndims>& block_shape,
       const ac::shape<ndims>& block_offset, std::vector<T*>& outputs)
{
    const uint64_t block_nelems{prod(block_shape)};
    for (uint64_t i{0}; i < block_nelems; ++i) {
        for (size_t j{0}; j < outputs.size(); ++j) {

            // Block coords
            const ac::shape<ndims> block_coords{to_spatial(i, block_shape)};

            // Input coords
            const ac::shape<ndims> in_coords{block_offset + block_coords};

            const uint64_t in_idx{to_linear(in_coords, mm)};
            ERRCHK(in_idx < prod(mm));

            outputs[j][in_idx] = input[i + j * block_nelems];
        }
    }
}

/**
 * Forwards declarations (For user types)
 */
template void pack<UserType, UserNdims>(const UserShape& mm, const UserShape& block_shape,
                                        const UserIndex& block_offset,
                                        const std::vector<UserType*>& inputs, UserType* output);

template void unpack(const UserType* input, const UserShape& mm, const UserShape& block_shape,
                     const UserShape& block_offset, std::vector<UserType*>& outputs);

void
test_pack(void)
{
    const size_t count{10};
    const size_t rr{1};
    Buffer<uint64_t, HostMemoryResource> hin(count);
    Buffer<uint64_t, DeviceMemoryResource> din(count);
    Buffer<uint64_t, DeviceMemoryResource> dout(count - 2 * rr);
    Buffer<uint64_t, HostMemoryResource> hout(count - 2 * rr);
    std::iota(hin.begin(), hin.end(), 0);
    std::fill(hout.begin(), hout.end(), 0);
    // ac::copy(hin.begin(), hin.end(), din.begin());
    migrate(hin, din);

    ac::shape<1> mm{count};
    ac::shape<1> block_shape{count - 2 * rr};
    ac::shape<1> block_offset{rr};
    std::vector<uint64_t*> inputs{din.data()};
    pack(mm, block_shape, block_offset, inputs, dout.data());
    migrate(dout, hout);
    // ac::copy(dout.begin(), dout.end(), hout.begin());

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
