#include "pack.h"

#include <numeric>

template <typename T, size_t N>
void
pack(const Shape& mm, const Shape& block_shape, const Index& block_offset,
     const ac::array<T*, N>& inputs, ac::host_vector<T>& output)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {

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

template <typename T, size_t N>
void
unpack(const ac::host_vector<T>& input, const Shape& mm, const Shape& block_shape,
       const Index& block_offset, ac::array<T*, N>& outputs)
{
    const uint64_t block_nelems = prod(block_shape);
    for (uint64_t i = 0; i < block_nelems; ++i) {
        for (size_t j = 0; j < outputs.size(); ++j) {

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

template void pack<AcReal, PACK_MAX_INPUTS>(const Shape&, const Shape&, const Index&,
                                            const ac::array<AcReal*, PACK_MAX_INPUTS>&,
                                            ac::host_vector<AcReal>&);

template void unpack<AcReal, PACK_MAX_INPUTS>(const ac::host_vector<AcReal>&, const Shape&,
                                              const Shape&, const Index&,
                                              ac::array<AcReal*, PACK_MAX_INPUTS>&);

template void pack<uint64_t, PACK_MAX_INPUTS>(const Shape&, const Shape&, const Index&,
                                              const std::array<uint64_t*, PACK_MAX_INPUTS>&,
                                              ac::host_vector<uint64_t>&);

template void unpack<uint64_t, PACK_MAX_INPUTS>(const ac::host_vector<uint64_t>&, const Shape&,
                                                const Shape&, const Index&,
                                                std::array<uint64_t*, PACK_MAX_INPUTS>&);

void
test_pack(void)
{
    const size_t count = 10;
    const size_t rr    = 1;
    ac::host_vector<uint64_t> hin(count);
    ac::device_vector<uint64_t> din(count);
    ac::device_vector<uint64_t> dout(count - 2 * rr);
    ac::host_vector<uint64_t> hout(count - 2 * rr);
    std::iota(hin.begin(), hin.end(), 0);
    std::fill(hout.begin(), hout.end(), 0);
    ac::copy(hin.begin(), hin.end(), din.begin());

    Shape mm{count};
    Shape block_shape{count - 2 * rr};
    Index block_offset{rr};
    ac::array<uint64_t*, PACK_MAX_INPUTS> inputs{din.data()};
    pack(mm, block_shape, block_offset, inputs, dout);
    ac::copy(dout.begin(), dout.end(), hout.begin());

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
