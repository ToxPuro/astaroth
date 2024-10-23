#include "static_array.h"

#include "print_debug.h"
#include "type_conversion.h"

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
        StaticArray<uint64_t, 3> a = {1, 2, 3};
        ERRCHK(a[0] == 1);
        ERRCHK(a[1] == 2);
        ERRCHK(a[2] == 3);
        // PRINT_DEBUG(a);
    }
    {
        StaticArray<uint64_t, 3> a = {1, 2, 3};
        StaticArray<uint64_t, 3> b = {4, 5, 6};
        // PRINT_DEBUG(a + b);
    }
    {
        StaticArray<uint64_t, 3> a = {1, 2, 3};
        StaticArray<uint64_t, 3> b = {4, 5, 6};
        StaticArray<uint64_t, 3> c = {1, 1, 1};
        // PRINT_DEBUG(as<uint64_t>(2) * a + b / c);
    }
    {
        StaticArray<uint64_t, 3> a(3, 1);
        StaticArray<uint64_t, 3> b(3, 1);
        PRINT_DEBUG(a);
        PRINT_DEBUG(b);
        PRINT_DEBUG(a.dot(b));
    }
}
