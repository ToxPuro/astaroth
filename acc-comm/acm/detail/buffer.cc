#include "buffer.h"

void
test_buffer(void)
{
    const ac::buffer<double, ac::mr::host_allocator> vec(10);
    ERRCHK(vec.size() == 10);

    const auto vec1 = ac::buffer<int, ac::mr::pinned_host_allocator>(20, -1);
    ERRCHK(vec1.size() == 20);
    ERRCHK(vec1[0] == -1);
    ERRCHK(vec1[vec1.size() - 1] == -1);

    // Initializer list constructor disabled to avoid confusion
    // use std::fill, std::iota, and other functions to initialize
    // the buffer after creation
    // const ac::buffer<uint64_t> vec2{1, 2, 3, 4};
    // ERRCHK(vec2[0] == 1);
    // ERRCHK(vec2[1] == 2);
    // ERRCHK(vec2[2] == 3);
    // ERRCHK(vec2[3] == 4);

    // TODO check for memory leaks
    PRINT_LOG_WARNING("TODO: check for memory leaks");
    ac::host_buffer<int> in{10};
    auto                 dbuf{in.to_device()};
    auto                 out{dbuf.to_host()};

    PRINT_LOG_INFO("OK");
}
