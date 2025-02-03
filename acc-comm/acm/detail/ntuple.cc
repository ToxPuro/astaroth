#include "ntuple.h"

#include "errchk.h"
#include "type_conversion.h"

static void
test_fn(const ac::ntuple<uint64_t>& arr)
{
    ERRCHK((arr == ac::ntuple<uint64_t>{1, 2, 3}));
}

void
test_ntuple(void)
{

    {
        ac::ntuple<uint64_t> a{3};
        for (size_t i{0}; i < a.size(); ++i)
            ERRCHK(a[i] == 3);
        // PRINT_DEBUG(a);
    }
    {
        ac::ntuple<uint64_t> a{2, 5};
        ERRCHK(a[0] == 2);
        ERRCHK(a[1] == 5);
        // PRINT_DEBUG(a);
    }
    {
        ac::ntuple<uint64_t> a{3, 4, 5};
        ERRCHK((a == ac::ntuple<uint64_t>{3, 4, 5}));
    }
    {
        ac::ntuple<uint64_t> a{1, 2, 3};
        ac::ntuple<uint64_t> b{4, 5, 6};
        ac::ntuple<uint64_t> c{5, 7, 9};
        ERRCHK(a + b == c);
        // PRINT_DEBUG(a + b);
    }
    {
        ac::ntuple<uint64_t> a{1, 2, 3};
        ac::ntuple<uint64_t> b{4, 5, 6};
        ac::ntuple<uint64_t> c{2, 1, 9};
        ac::ntuple<uint64_t> d{2 * 1 + 4 / 2, 2 * 2 + 5 / 1, 2 * 3 + 6 / 9};
        ERRCHK(as<uint64_t>(2) * a + b / c == d);
    }
    {
        ac::ntuple<uint64_t> a{8, 9, 10, 11};
        ac::ntuple<uint64_t> b{7, 6, 5, 4};
        ac::ntuple<uint64_t> c{1, 3, 0, 3};
        ERRCHK(a % b == c);
    }
    {
        test_fn(ac::ntuple<uint64_t>{1, 2, 3});
    }
    {
        const size_t    count       = 10;
        int             data[count] = {1};
        ac::ntuple<int> a{ac::make_ntuple_from_ptr(count, data)};
        ERRCHK(a[0] == 1);
        ERRCHK(a[9] == 0);
    }
    {
        ac::ntuple<int> a{ac::make_ntuple(100, 1)};
        ERRCHK(a[0] == 1);
        ERRCHK(a[99] == 1);
    }
    PRINT_LOG_INFO("OK");
}
