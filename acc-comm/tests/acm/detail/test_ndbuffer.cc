#include <cstdlib>
#include <numeric>

#include "acm/detail/ndbuffer.h"

int
main()
{
    {
        ac::ndbuffer<uint64_t, ac::mr::host_allocator> mesh(ac::shape{4, 4}, 0);
        ac::fill<uint64_t>(1, ac::shape{2, 2}, ac::shape{1, 1}, mesh);
        ERRCHK(mesh[0] == 0);
        ERRCHK(mesh[5] == 1);
        ERRCHK(mesh[10] == 1);
        ERRCHK(mesh[15] == 0);
    }
    {
        const ac::host_ndbuffer<double> vec(ac::shape{1, 2, 3});
        ERRCHK((vec.shape() == ac::shape{1, 2, 3}));
    }
    {
        const ac::shape              shape{8, 7, 6};
        const ac::host_ndbuffer<int> buf{shape, -1};
        ERRCHK(buf.size() == prod(shape));
        ERRCHK(std::all_of(buf.begin(), buf.end(), [](const auto elem) { return elem == -1; }));

        ac::host_ndbuffer<int> ref{shape};
        std::iota(ref.begin(), ref.end(), 1);

        const auto in{ref.copy()};
        const auto dbuf{in.to_device()};
        const auto out{dbuf.to_host()};
        ERRCHK(std::equal(in.begin(), in.end(), ref.begin()));
    }

    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
