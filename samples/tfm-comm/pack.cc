#include "pack.h"

#include <numeric>

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

    Shape<NDIMS> mm{count};
    Shape<NDIMS> block_shape{count - 2 * rr};
    Index<NDIMS> block_offset{rr};
    ac::array<uint64_t*, 1> inputs{ac::raw_pointer_cast(din.data())};
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
