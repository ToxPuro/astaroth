#include <cstdlib>
#include <numeric>

#include "acm/detail/buffer.h"
#include "acm/detail/pack.h"

#include "acm/detail/ndbuffer.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"
#include "acm/detail/timer.h"

#include "acm/detail/convert.h"

static void
test_pack(const ac::shape& nn, const ac::index& rr)
{
    const auto mm{nn + 2 * rr};

    // Partition and prune the domain
    // const auto segments{prune(partition(mm, nn, rr), nn, rr)};
    const auto segments{partition(mm, nn, rr)};

    // Allocate
    ac::host_ndbuffer<uint64_t>   hin{mm};
    ac::host_ndbuffer<uint64_t>   hout{mm};
    ac::device_ndbuffer<uint64_t> din{mm};
    ac::device_ndbuffer<uint64_t> dout{mm};

    std::vector<ac::device_buffer<uint64_t>> pack_buffers;
    for (const auto& segment : segments)
        pack_buffers.push_back(ac::device_buffer<uint64_t>{prod(segment.dims)});

    // Init
    std::iota(hin.begin(), hin.end(), 1);
    ac::copy(hin.get(), din.get());

    // Pack and unpack
    for (size_t i{0}; i < segments.size(); ++i)
        acm::pack(mm, segments[i].dims, segments[i].offset, {din.get()}, pack_buffers[i].get());

    for (size_t i{0}; i < segments.size(); ++i)
        acm::unpack(pack_buffers[i].get(), mm, segments[i].dims, segments[i].offset, {dout.get()});

    ac::copy(dout.get(), hout.get());
    ERRCHK(equals(hin.get(), hout.get()));

    // Benchmark
    // ac::timer t;
    // for (size_t i{0}; i < segments.size(); ++i)
    //     acm::pack(mm, segments[i].dims, segments[i].offset, {din.get()}, pack_buffers[i].get());

    // t.print_lap("Pack");

    // for (size_t i{0}; i < segments.size(); ++i)
    //     acm::unpack(pack_buffers[i].get(), mm, segments[i].dims, segments[i].offset,
    //     {dout.get()});

    // t.print_lap("Unpack");
}

static void
test_pack_batched(const ac::shape& nn, const ac::index& rr)
{
    const auto mm{nn + 2 * rr};

    // Partition and prune the domain
    // const auto segments{prune(partition(mm, nn, rr), nn, rr)};
    const auto segments{partition(mm, nn, rr)};

    // Allocate
    ac::host_ndbuffer<uint64_t>   hin{mm};
    ac::host_ndbuffer<uint64_t>   hout{mm};
    ac::device_ndbuffer<uint64_t> din{mm};
    ac::device_ndbuffer<uint64_t> dout{mm};

    std::vector<ac::device_buffer<uint64_t>> pack_buffers;
    for (const auto& segment : segments)
        pack_buffers.push_back(ac::device_buffer<uint64_t>{prod(segment.dims)});

    // Init
    std::iota(hin.begin(), hin.end(), 1);
    ac::copy(hin.get(), din.get());

    // Pack and unpack
    acm::pack_batched(mm, {din.get()}, segments, unwrap_get(pack_buffers));
    acm::unpack_batched(segments, unwrap_get(pack_buffers), mm, {dout.get()});

    ac::copy(dout.get(), hout.get());
    ERRCHK(equals(hin.get(), hout.get()));

    // Benchmmark
    // ac::timer t;
    // acm::pack_batched(mm, {din.get()}, segments, unwrap_get(pack_buffers));
    // t.print_lap("Batched pack");
    // acm::unpack_batched(segments, unwrap_get(pack_buffers), mm, {dout.get()});
    // t.print_lap("Batched unpack");
}

int
main()
{
    {
        const size_t                                   count{10};
        const size_t                                   rr{1};
        ac::buffer<uint64_t, ac::mr::host_allocator>   hin(count);
        ac::buffer<uint64_t, ac::mr::device_allocator> din(count);
        ac::buffer<uint64_t, ac::mr::device_allocator> dout(count - 2 * rr);
        ac::buffer<uint64_t, ac::mr::host_allocator>   hout(count - 2 * rr);
        std::iota(hin.begin(), hin.end(), 0);
        std::fill(hout.begin(), hout.end(), 0);
        // ac::copy(hin.begin(), hin.end(), din.begin());
        migrate(hin, din);

        ac::shape                              mm{count};
        ac::shape                              block_shape{count - 2 * rr};
        ac::shape                              block_offset{rr};
        std::vector<ac::device_view<uint64_t>> inputs{
            ac::device_view<uint64_t>{din.size(), din.data()}};
        acm::pack(mm,
                  block_shape,
                  block_offset,
                  inputs,
                  ac::device_view<uint64_t>{dout.size(), dout.data()});
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
        // ac::buffer<double, ac::mr::host_allocator> a(10, 0);
        // ac::buffer<double, ac::mr::host_allocator> b(10, 1);
        // ac::buffer<double, ac::mr::host_allocator> c(10, 2);
        // std::vector<ac::host_view<double>> d{ac::host_view<double>{a.size(),
        // a.data()},
        //                                         ac::host_view<double>{b.size(), b.data()},
        //                                         ac::host_view<double>{c.size(),
        //                                         c.data()}};
        // std::cout << a << std::endl;
        // std::cout << *d[1] << std::endl;
        // std::cout << "-----------------" << std::endl;
    }
    {
        std::vector<std::tuple<ac::shape, ac::index>> inputs{
            // std::tuple<ac::shape, ac::index>{ac::shape{128, 128, 128}, ac::index{3, 3, 3}},
            // std::tuple<ac::shape, ac::index>{ac::shape{256, 256, 256}, ac::index{3, 3, 3}},
            // std::tuple<ac::shape, ac::index>{ac::shape{256, 256, 256}, ac::index{6, 6, 6}},
            //
            std::tuple<ac::shape, ac::index>{ac::shape{8}, ac::index{3}},
            std::tuple<ac::shape, ac::index>{ac::shape{4, 4}, ac::index{2, 2}},
            std::tuple<ac::shape, ac::index>{ac::shape{4, 8}, ac::index{2, 2}},
            std::tuple<ac::shape, ac::index>{ac::shape{8, 6, 4}, ac::index{2, 2, 2}},
            // std::tuple<ac::shape, ac::index>{ac::shape{8, 6, 4, 4}, ac::index{2, 2, 2, 2}},
            //
            // std::tuple<ac::shape, ac::index>{ac::shape{4, 5, 6, 7, 8}, ac::index{2, 2, 3, 3, 3}},
            // std::tuple<ac::shape, ac::index>{ac::shape{2, 4, 6, 8, 6, 4, 2},
            //                                  ac::index{1, 2, 3, 4, 3, 2, 1}},
        };
        for (const auto& input : inputs)
            test_pack(std::get<0>(input), std::get<1>(input));

        for (const auto& input : inputs)
            test_pack_batched(std::get<0>(input), std::get<1>(input));
    }
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
