#include "pack.h"
#include "pack_batched.h"

#include <numeric>

#include "ndbuffer.h"
#include <algorithm>
#include <memory>

class TestClass {
  private:
    std::shared_ptr<int> resource;

  public:
    TestClass(const size_t count)
        : resource{std::make_shared<int>(count)}
    {
    }
};

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
        std::vector<ac::mr::device_ptr<uint64_t>> inputs{
            ac::mr::device_ptr<uint64_t>{din.size(), din.data()}};
        pack(mm,
             block_shape,
             block_offset,
             inputs,
             ac::mr::device_ptr<uint64_t>{dout.size(), dout.data()});
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
        // std::vector<ac::mr::host_ptr<double>> d{ac::mr::host_ptr<double>{a.size(), a.data()},
        //                                         ac::mr::host_ptr<double>{b.size(), b.data()},
        //                                         ac::mr::host_ptr<double>{c.size(), c.data()}};
        // std::cout << a << std::endl;
        // std::cout << *d[1] << std::endl;
        // std::cout << "-----------------" << std::endl;
    }
    // {
    //     const size_t count{10};
    //     const size_t rr{1};
    //     ac::buffer<uint64_t, ac::mr::host_memory_resource> hin(count);
    //     ac::buffer<uint64_t, ac::mr::device_memory_resource> din(count);
    //     ac::buffer<uint64_t, ac::mr::device_memory_resource> din2(count);
    //     ac::buffer<uint64_t, ac::mr::device_memory_resource> dout2(count - 2 * rr);
    //     ac::buffer<uint64_t, ac::mr::device_memory_resource> dout(count - 2 * rr);
    //     ac::buffer<uint64_t, ac::mr::host_memory_resource> hout(count - 2 * rr);
    //     std::iota(hin.begin(), hin.end(), 0);
    //     std::fill(hout.begin(), hout.end(), 0);
    //     // ac::copy(hin.begin(), hin.end(), din.begin());
    //     migrate(hin, din);

    //     Shape mm{count};
    //     Shape block_shape{count - 2 * rr};
    //     Shape block_offset{rr};
    //     std::vector<ac::mr::device_ptr<uint64_t>> inputs{
    //         ac::mr::device_ptr<uint64_t>{din.size(), din.data()}};

    //     auto output{std::vector<ac::mr::device_ptr<uint64_t>>{
    //         ac::mr::device_ptr<uint64_t>{dout.size(), dout.data()}}};
    //     ac::segment segment{block_shape, block_offset};
    //     pack_batched(mm, inputs, std::vector<ac::segment>{segment}, output);
    //     migrate(dout, hout);
    //     // ac::copy(dout.begin(), dout.end(), hout.begin());

    //     // hout.display();
    //     ERRCHK(hout[0] == 1);
    //     ERRCHK(hout[1] == 2);
    //     ERRCHK(hout[2] == 3);
    //     ERRCHK(hout[3] == 4);
    //     ERRCHK(hout[4] == 5);
    //     ERRCHK(hout[5] == 6);
    //     ERRCHK(hout[6] == 7);
    //     ERRCHK(hout[7] == 8);
    // }

    {
        using device_buffer = ac::ndbuffer<uint64_t, ac::mr::device_memory_resource>;
        using host_buffer   = ac::ndbuffer<uint64_t, ac::mr::host_memory_resource>;

        const Shape nn{10};
        const Shape rr{2};
        const Shape mm{2 * rr + nn};

        std::cout << "Generating..." << std::endl;

        constexpr size_t FIELD_COUNT(4);
        std::vector<host_buffer> hbuffers;
        std::vector<device_buffer> dbuffers;
        for (size_t i{0}; i < FIELD_COUNT; ++i) {
            hbuffers.push_back(host_buffer{mm});
            std::iota(hbuffers[i].begin(), hbuffers[i].end(), 0);
            hbuffers[i].display();

            dbuffers.push_back(device_buffer{mm});
            migrate(hbuffers[i].buffer, dbuffers[i].buffer);
        }

        std::vector<ac::segment> segments;
        segments.push_back(ac::segment{rr, Index{0}});
        segments.push_back(ac::segment{rr, mm - rr});

        std::cout << "Complete" << std::endl;
        // vec.push_back(TestClass{10});
        // std::vector<ac::buffer<uint64_t, ac::mr::host_memory_resource>> hbuffers{
        //     ac::buffer<uint64_t, ac::mr::host_memory_resource>{10}};
    }
}
