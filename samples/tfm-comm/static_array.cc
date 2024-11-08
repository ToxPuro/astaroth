#include "static_array.h"

#include "print_debug.h"
#include "type_conversion.h"

static void
test_fn(StaticArray<uint64_t, 3> arr)
{
    ERRCHK((arr == StaticArray<uint64_t, 3>{1, 2, 3}));
}

void
test_static_array(void)
{

    {
        StaticArray<uint64_t, 4> a(3);
        for (size_t i = 0; i < a.count; ++i)
            ERRCHK(a[i] == 0);
        // PRINT_DEBUG(a);
    }
    {
        StaticArray<uint64_t, 4> a(2, 5);
        for (size_t i = 0; i < a.count; ++i)
            ERRCHK(a[i] == 5);
        // PRINT_DEBUG(a);
    }
    {
        StaticArray<uint64_t, 3> a{3, 4, 5};
        ERRCHK((a == StaticArray<uint64_t, 3>{3, 4, 5}));
    }
    {
        StaticArray<uint64_t, 3> a{1, 2, 3};
        StaticArray<uint64_t, 3> b{4, 5, 6};
        StaticArray<uint64_t, 3> c{5, 7, 9};
        ERRCHK(a + b == c);
        // PRINT_DEBUG(a + b);
    }
    {
        StaticArray<uint64_t, 3> a{1, 2, 3};
        StaticArray<uint64_t, 3> b{4, 5, 6};
        StaticArray<uint64_t, 3> c{2, 1, 9};
        StaticArray<uint64_t, 3> d{2 * 1 + 4 / 2, 2 * 2 + 5 / 1, 2 * 3 + 6 / 9};
        ERRCHK(as<uint64_t>(2) * a + b / c == d);
    }
    {
        StaticArray<uint64_t, 50> a{8, 9, 10, 11};
        StaticArray<uint64_t, 50> b{7, 6, 5, 4};
        StaticArray<uint64_t, 50> c{1, 3, 0, 3};
        ERRCHK(a % b == c);
    }
    {
        test_fn(StaticArray<uint64_t, 3>{1, 2, 3});
    }
}
