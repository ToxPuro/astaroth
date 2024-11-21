#include "buffer.h"

void
test_buffer(void)
{
    const ac::buffer<double, HostMemoryResource> vec(10);
    ERRCHK(vec.size() == 10);

    const auto vec1 = ac::buffer<int, PinnedHostMemoryResource>(20, -1);
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
}
