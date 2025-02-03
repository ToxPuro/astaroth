#include "pack.h"

#include <numeric>

#include "convert.h"
#include "ndbuffer.h"
#include <algorithm>
#include <memory>

void
test_pack(void)
{
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

        Shape mm{count};
        Shape block_shape{count - 2 * rr};
        Shape block_offset{rr};
        std::vector<ac::mr::device_pointer<uint64_t>> inputs{
            ac::mr::device_pointer<uint64_t>{din.size(), din.data()}};
        pack(mm,
             block_shape,
             block_offset,
             inputs,
             ac::mr::device_pointer<uint64_t>{dout.size(), dout.data()});
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
        // std::cout << hout << std::endl;
        // ac::buffer<double, ac::mr::host_memory_resource> a(10, 0);
        // ac::buffer<double, ac::mr::host_memory_resource> b(10, 1);
        // ac::buffer<double, ac::mr::host_memory_resource> c(10, 2);
        // std::vector<ac::mr::host_pointer<double>> d{ac::mr::host_pointer<double>{a.size(),
        // a.data()},
        //                                         ac::mr::host_pointer<double>{b.size(), b.data()},
        //                                         ac::mr::host_pointer<double>{c.size(),
        //                                         c.data()}};
        // std::cout << a << std::endl;
        // std::cout << *d[1] << std::endl;
        // std::cout << "-----------------" << std::endl;
    }
}
