#include <cstdlib>

#include "acm/detail/memory.h"

int
main()
{
    constexpr size_t   count{10};
    auto               in_data  = new int[count];
    auto               out_data = new int[count];
    ac::host_view<int> in{count, in_data};
    ac::host_view<int> out{count, out_data};

    ac::copy(in, out);
    ERRCHK(equals(in, out));

    delete[] in_data;
    delete[] out_data;
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
