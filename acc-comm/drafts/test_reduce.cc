#include <cstdlib>
#include <numeric>

#include "acm/detail/ndbuffer.h"
#include "acm/detail/reduce.h"

int
main()
{
    {
        ac::shape mm{4, 4};
        ac::shape nn{2, 2};
        ac::index rr{1, 1};

        ac::host_ndbuffer<uint64_t> h0{mm};
        ac::host_ndbuffer<uint64_t> h1{mm};
        ac::host_ndbuffer<uint64_t> h2{mm};

        std::iota(h0.begin(), h0.end(), 1);
        std::iota(h1.begin(), h1.end(), 1 + h0.size());
        std::iota(h2.begin(), h2.end(), 1 + h0.size() + h1.size());

        h0.display();
        h1.display();
        h2.display();

        auto d0{h0.to_device()};
        auto d1{h1.to_device()};
        auto d2{h2.to_device()};

        // const auto                                    count{h0.size() + h1.size() + h2.size()};
        std::vector<ac::device_view<uint64_t>> inputs{d0.get(), d1.get(), d2.get()};
        ac::device_buffer<uint64_t>            output{inputs.size()};
        ac::segmented_reduce(mm, nn, rr, inputs, output.get());

        auto host_output{output.to_host()};
        ERRCHK(host_output.size() == inputs.size());
        host_output.display();

        ERRCHK(host_output[0] == 6 + 7 + 10 + 11);
        ERRCHK(host_output[1] == 22 + 23 + 26 + 27);
        ERRCHK(host_output[2] == 38 + 39 + 42 + 43);
    }

    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
