#include "algorithm.h"
#include <numeric>

#include "errchk_print.h"

#include "ndbuffer.h"

#include "pack.h"
#include "type_conversion.h"

void
test_algorithm()
{
    {
        const ac::shape        nn{4, 3};
        const auto             rr{ac::make_index(nn.size(), 1)};
        const auto             mm{nn + 2 * rr};
        ac::host_ndbuffer<int> ref{mm};
        ac::host_ndbuffer<int> pack_buffer{nn};
        ac::host_ndbuffer<int> tst{nn};

        // Test pack
        std::iota(pack_buffer.begin(), pack_buffer.end(), 1);
        unpack(pack_buffer.get(), mm, nn, rr, {ref.get()});

        std::fill(pack_buffer.begin(), pack_buffer.end(), -1);
        pack(mm, nn, rr, {ref.get()}, pack_buffer.get());
        for (size_t i{0}; i < pack_buffer.size(); ++i)
            ERRCHK(pack_buffer[i] == as<int>(i) + 1);

        // Test transform
        ac::transform(
            pack_buffer.get(), [](const auto& elem) { return elem * elem; }, tst.get());
        for (size_t i{0}; i < tst.size(); ++i)
            ERRCHK(tst[i] == as<int>((i + 1) * (i + 1)));

        // Test reduce
        ac::segmented_reduce(
            1,
            prod(nn),
            pack_buffer.get(),
            [](const auto& a, const auto& b) { return a + b; },
            0,
            tst.get());
        ERRCHK(tst[0] == (prod(nn) * (prod(nn) + 1)) / 2);
    }

    {
        const ac::shape        nn{4, 4};
        const auto             rr{ac::make_index(nn.size(), 1)};
        const auto             mm{nn + 2 * rr};
        ac::host_ndbuffer<int> ref{mm};
        ac::host_ndbuffer<int> tst{ac::concat(nn, static_cast<uint64_t>(2))};
        std::iota(ref.begin(), ref.end(), 1);

        // Pack
        std::vector pack_inputs{ref.get(), ref.get()};
        pack(mm, nn, rr, pack_inputs, tst.get());
        ref.display();
        tst.display();

        // Transform (square)
        ac::transform(
            tst.get(), [](const int& a) { return a * a; }, tst.get());
        tst.display();

        // Reduce
        const size_t         num_segments{pack_inputs.size()};
        const size_t         stride{prod(nn)};
        ac::host_buffer<int> reduce_buffer{num_segments};
        ac::segmented_reduce(
            num_segments,
            stride,
            tst.get(),
            [](const auto& a, const auto& b) { return a + b; },
            0,
            reduce_buffer.get());
        std::cout << "reduce-buffer" << std::endl;
        reduce_buffer.display();

        // Transform (average)
        ac::transform(
            reduce_buffer.get(),
            [&stride](const int& a) { return a / as<int>(stride); },
            reduce_buffer.get());
        reduce_buffer.display();
    }

    {
        const ac::shape        nn{4, 4};
        const auto             rr{ac::make_index(nn.size(), 1)};
        const auto             mm{nn + 2 * rr};
        ac::host_ndbuffer<int> ref{mm};
        std::vector            pack_inputs{ref.get(), ref.get()};
        ac::host_ndbuffer<int> tst{ac::concat(nn, as<uint64_t>(pack_inputs.size()))};
        std::iota(ref.begin(), ref.end(), 1);

        // Pack
        pack(mm, nn, rr, pack_inputs, tst.get());
        ref.display();
        tst.display();

        // Transform (square)
        ac::transform(
            tst.get(),
            tst.get(),
            tst.get(),
            [](const int& a, const int& b, const int& c) { return a * b + c; },
            tst.get());
        tst.display();

        // Reduce
        const size_t num_segments{pack_inputs.size()};
        const size_t stride{prod(nn)};
        ERRCHK(stride * num_segments == tst.size());

        ac::host_buffer<int> reduce_buffer{num_segments};
        ac::segmented_reduce(
            num_segments,
            stride,
            tst.get(),
            [](const auto& a, const auto& b) { return a + b; },
            0,
            reduce_buffer.get());
        std::cout << "reduce-buffer raw" << std::endl;
        reduce_buffer.display();

        // Transform (average)
        ac::transform(
            reduce_buffer.get(),
            [&stride](const int& a) { return a / as<int>(stride); },
            reduce_buffer.get());
        std::cout << "reduce-buffer avg" << std::endl;
        reduce_buffer.display();
    }

    // ERRCHK(std::equal(ref.begin(), ref.end(), tst.begin()));

    PRINT_LOG_WARNING("Not implemented");
}
