#include "datatypes.h"

#include "print_debug.h"
#include "type_conversion.h"

static void
test_fn(ac::array<uint64_t, 3> arr)
{
    ERRCHK((arr == ac::array<uint64_t, 3>{1, 2, 3}));
}

void
test_datatypes(void)
{

    {
        ac::array<uint64_t, 1> a{3};
        for (size_t i{0}; i < a.size(); ++i)
            ERRCHK(a[i] == 3);
        // PRINT_DEBUG(a);
    }
    {
        ac::array<uint64_t, 2> a{2, 5};
        ERRCHK(a[0] == 2);
        ERRCHK(a[1] == 5);
        // PRINT_DEBUG(a);
    }
    {
        ac::array<uint64_t, 3> a{3, 4, 5};
        ERRCHK((a == ac::array<uint64_t, 3>{3, 4, 5}));
    }
    {
        ac::array<uint64_t, 3> a{1, 2, 3};
        ac::array<uint64_t, 3> b{4, 5, 6};
        ac::array<uint64_t, 3> c{5, 7, 9};
        ERRCHK(a + b == c);
        // PRINT_DEBUG(a + b);
    }
    {
        ac::array<uint64_t, 3> a{1, 2, 3};
        ac::array<uint64_t, 3> b{4, 5, 6};
        ac::array<uint64_t, 3> c{2, 1, 9};
        ac::array<uint64_t, 3> d{2 * 1 + 4 / 2, 2 * 2 + 5 / 1, 2 * 3 + 6 / 9};
        ERRCHK(as<uint64_t>(2) * a + b / c == d);
    }
    {
        ac::array<uint64_t, 4> a{8, 9, 10, 11};
        ac::array<uint64_t, 4> b{7, 6, 5, 4};
        ac::array<uint64_t, 4> c{1, 3, 0, 3};
        ERRCHK(a % b == c);
    }
    {
        test_fn(ac::array<uint64_t, 3>{1, 2, 3});
    }
}
