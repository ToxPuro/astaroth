#include "pack.h"
#include "pack_batched.h"

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

    {
        using device_buffer   = ac::buffer<uint64_t, ac::mr::device_memory_resource>;
        using device_ndbuffer = ac::ndbuffer<uint64_t, ac::mr::device_memory_resource>;
        using host_buffer     = ac::ndbuffer<uint64_t, ac::mr::host_memory_resource>;

        const Shape nn{10};
        const Shape rr{2};
        const Shape mm{2 * rr + nn};

        std::cout << "Generating..." << std::endl;

        constexpr size_t FIELD_COUNT(4);
        std::vector<host_buffer> hbuffers;
        std::vector<device_ndbuffer> dbuffers;
        for (size_t i{0}; i < FIELD_COUNT; ++i) {
            hbuffers.emplace_back(mm);
            std::iota(hbuffers[i].begin(), hbuffers[i].end(), i * prod(mm));
            hbuffers[i].display();

            dbuffers.emplace_back(mm);
            migrate(hbuffers[i].buffer, dbuffers[i].buffer);
        }

        std::vector<ac::segment> segments;
        segments.emplace_back(rr, Index{0});
        segments.emplace_back(nn, rr);
        segments.emplace_back(rr, mm - rr);

        std::vector<device_buffer> segment_buffers;
        for (size_t i{0}; i < segments.size(); ++i)
            segment_buffers.emplace_back(dbuffers.size() * prod(segments[i].dims));

        pack_batched_host(mm, ac::unwrap(hbuffers), segments, ac::unwrap(segment_buffers));

        for (const auto& buf : segment_buffers)
            buf.display();

        std::vector<uint64_t> lmodel{0, 1, 14, 15, 28, 29, 42, 43};
        std::vector<uint64_t> cmodel{2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                                     20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37,
                                     38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53};
        std::vector<uint64_t> rmodel{12, 13, 26, 27, 40, 41, 54, 55};

        ERRCHK(segment_buffers[0].size() == lmodel.size());
        for (size_t i{0}; i < segment_buffers[0].size(); ++i)
            ERRCHK(segment_buffers[0][i] == lmodel[i]);

        ERRCHK(segment_buffers[1].size() == cmodel.size());
        for (size_t i{0}; i < segment_buffers[0].size(); ++i)
            ERRCHK(segment_buffers[1][i] == cmodel[i]);

        ERRCHK(segment_buffers[2].size() == rmodel.size());
        for (size_t i{0}; i < segment_buffers[0].size(); ++i)
            ERRCHK(segment_buffers[2][i] == rmodel[i]);

        for (auto& buf : hbuffers)
            std::fill_n(buf.begin(), buf.size(), 0);

        unpack_batched_host(segments, ac::unwrap(segment_buffers), mm, ac::unwrap(hbuffers));

        for (size_t i{0}; i < hbuffers.size(); ++i) {
            hbuffers[i].display();
            for (size_t j{0}; j < hbuffers[i].buffer.size(); ++j)
                ERRCHK(hbuffers[i].buffer[j] == j + i * hbuffers[i].buffer.size());
        }

        std::cout << "Complete" << std::endl;
    }
}
