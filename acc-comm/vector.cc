#include "vector.h"

#include "errchk.h"
#include "type_conversion.h"

static void
test_fn(const std::vector<uint64_t>& arr)
{
    ERRCHK((arr == std::vector<uint64_t>{1, 2, 3}));
}

void
test_vector(void)
{

    {
        std::vector<uint64_t> a{3};
        for (size_t i{0}; i < a.size(); ++i)
            ERRCHK(a[i] == 3);
        // PRINT_DEBUG(a);
    }
    {
        std::vector<uint64_t> a{2, 5};
        ERRCHK(a[0] == 2);
        ERRCHK(a[1] == 5);
        // PRINT_DEBUG(a);
    }
    {
        std::vector<uint64_t> a{3, 4, 5};
        ERRCHK((a == std::vector<uint64_t>{3, 4, 5}));
    }
    {
        std::vector<uint64_t> a{1, 2, 3};
        std::vector<uint64_t> b{4, 5, 6};
        std::vector<uint64_t> c{5, 7, 9};
        ERRCHK(a + b == c);
        // PRINT_DEBUG(a + b);
    }
    {
        std::vector<uint64_t> a{1, 2, 3};
        std::vector<uint64_t> b{4, 5, 6};
        std::vector<uint64_t> c{2, 1, 9};
        std::vector<uint64_t> d{2 * 1 + 4 / 2, 2 * 2 + 5 / 1, 2 * 3 + 6 / 9};
        ERRCHK(as<uint64_t>(2) * a + b / c == d);
    }
    {
        std::vector<uint64_t> a{8, 9, 10, 11};
        std::vector<uint64_t> b{7, 6, 5, 4};
        std::vector<uint64_t> c{1, 3, 0, 3};
        ERRCHK(a % b == c);
    }
    {
        test_fn(std::vector<uint64_t>{1, 2, 3});
    }
}
