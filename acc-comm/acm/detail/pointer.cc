#include <cstdlib>

#include "errchk.h"
#include "pointer.h"

static bool
equals(const ac::mr::host_pointer<int>& a, const ac::mr::host_pointer<int>& b)
{
    if (a.size() != b.size())
        return false;

    for (size_t i{0}; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

void
test_pointer()
{
    ac::mr::host_pointer<int> a{0, nullptr};
    ac::mr::device_pointer<int> b{0, nullptr};

    constexpr size_t count{10};
    auto in_data  = new int[count];
    auto out_data = new int[count];
    ac::mr::host_pointer<int> in{count, in_data};
    ac::mr::host_pointer<int> out{count, out_data};

    ac::mr::copy(in, out);
    ERRCHK(equals(in, out));

    delete[] in_data;
    delete[] out_data;
    PRINT_LOG_INFO("OK");
}
