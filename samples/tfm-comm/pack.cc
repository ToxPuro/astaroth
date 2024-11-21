#include "pack.h"

#include <numeric>

void
test_pack(void)
{
    const size_t count{10};
    const size_t rr{1};
    ac::vector<uint64_t, HostMemoryResource> hin(count);
    ac::vector<uint64_t, DeviceMemoryResource> din(count);
    ac::vector<uint64_t, DeviceMemoryResource> dout(count - 2 * rr);
    ac::vector<uint64_t, HostMemoryResource> hout(count - 2 * rr);
    std::iota(hin.begin(), hin.end(), 0);
    std::fill(hout.begin(), hout.end(), 0);
    // ac::copy(hin.begin(), hin.end(), din.begin());
    migrate(hin, din);

    ac::shape<1> mm{count};
    ac::shape<1> block_shape{count - 2 * rr};
    ac::shape<1> block_offset{rr};
    std::vector<ac::vector<uint64_t, DeviceMemoryResource>*> inputs{&din};
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

    // std::cout << "-------PACK------" << std::endl;
    // ac::vector<double> a(10, 0);
    // ac::vector<double> b(10, 1);
    // ac::vector<double> c(10, 2);
    // std::vector<ac::vector<double>*> d{&a, &b, &c};
    // std::cout << a << std::endl;
    // std::cout << *d[1] << std::endl;
    // std::cout << "-----------------" << std::endl;
}
