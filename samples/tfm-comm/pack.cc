#include "pack.h"

#include <numeric>

void
test_pack(void)
{
    const size_t count = 10;
    const size_t rr    = 1;
    Buffer<uint64_t, HostMemoryResource> hin(count);
    Buffer<uint64_t, DeviceMemoryResource> din(count);
    Buffer<uint64_t, DeviceMemoryResource> dout(count - 2 * rr);
    Buffer<uint64_t, HostMemoryResource> hout(count - 2 * rr);
    std::iota(hin.begin(), hin.end(), 0);
    std::fill(hout.begin(), hout.end(), 0);
    // ac::copy(hin.begin(), hin.end(), din.begin());
    migrate(hin, din);

    Shape<NDIMS> mm{count};
    Shape<NDIMS> block_shape{count - 2 * rr};
    Index<NDIMS> block_offset{rr};
    ac::array<uint64_t*, 1> inputs{din.data()};
    pack(mm, block_shape, block_offset, inputs, dout);
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
