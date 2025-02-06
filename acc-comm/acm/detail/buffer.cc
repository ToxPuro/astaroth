#include "buffer.h"

#include <numeric>

void
test_buffer(void)
{
    const ac::host_buffer<double> vec(10);
    ERRCHK(vec.size() == 10);

    const ac::host_buffer<int> buf{20, -1};
    ERRCHK(buf.size() == 20);
    ERRCHK(std::all_of(buf.begin(), buf.end(), [](const auto elem) { return elem == -1; }));

    ac::host_buffer<int> ref{10};
    std::iota(ref.begin(), ref.end(), 1);

    const auto in{ref.copy()};
    const auto dbuf{in.to_device()};
    const auto out{dbuf.to_host()};
    ERRCHK(std::equal(in.begin(), in.end(), ref.begin()));

    PRINT_LOG_INFO("OK");
}
