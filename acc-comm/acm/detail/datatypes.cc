#include "datatypes.h"

#include "print_debug.h"
#include "type_conversion.h"

#include "ntuple.h"

Dims
make_dims(const size_t count, const UserDatatype& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

static void
test_fn(ac::ntuple<uint64_t> arr)
{
    ERRCHK((arr == ac::ntuple<uint64_t>{1, 2, 3}));
}

static void
test_ntuple_static_and_non_static()
{
    ac::ntuple<int>           a{1, 2, 3};
    ac::static_ntuple<int, 4> b{a};
    ac::static_ntuple<int, 5> c{1, 2, 3, 4, 5};
    ac::ntuple<int>           d{b};
    ERRCHK(a == d);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << 2 * c + c / 3 << std::endl;
    ERRCHK((2 * c + c / 3 == ac::static_ntuple<int, 5>{2 * 1 + 1 / 3,
                                                       2 * 2 + 2 / 3,
                                                       2 * 3 + 3 / 3,
                                                       2 * 4 + 4 / 3,
                                                       2 * 5 + 5 / 3}));
}

void
test_datatypes(void)
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
        test_ntuple_static_and_non_static();
    }
    PRINT_LOG_INFO("OK");
}
