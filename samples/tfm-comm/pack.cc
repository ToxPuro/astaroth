#include "pack.h"

#include <numeric>

void
test_pack(void)
{
    const size_t count{10};
    const size_t rr{1};
    ac::buffer<uint64_t, ac::mr::host_memory_resource> hin(count);
    ac::buffer<uint64_t, ac::mr::device_memory_resource> din(count);
    ac::buffer<uint64_t, ac::mr::device_memory_resource> dout(count - 2 * rr);
    ac::buffer<uint64_t, ac::mr::host_memory_resource> hout(count - 2 * rr);
    std::iota(hin.begin(), hin.end(), 0);
    std::fill(hout.begin(), hout.end(), 0);
    // ac::copy(hin.begin(), hin.end(), din.begin());
    migrate(hin, din);

    ac::shape<1> mm{count};
    ac::shape<1> block_shape{count - 2 * rr};
    ac::shape<1> block_offset{rr};
    std::vector<ac::buffer<uint64_t, ac::mr::device_memory_resource>*> inputs{&din};
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
    // ac::buffer<double> a(10, 0);
    // ac::buffer<double> b(10, 1);
    // ac::buffer<double> c(10, 2);
    // std::vector<ac::buffer<double>*> d{&a, &b, &c};
    // std::cout << a << std::endl;
    // std::cout << *d[1] << std::endl;
    // std::cout << "-----------------" << std::endl;
}
