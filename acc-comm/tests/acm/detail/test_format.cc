#include <cstdlib>
#include <fstream>

#include "acm/detail/errchk.h"
#include "acm/detail/print_debug.h"

#include "acm/detail/experimental/fmt.h"

int
main()
{
    auto a{1.23456789101112131415161718};
    auto b{10lu};
    auto c{-1};
    auto d{9.876543210f};

    ERRCHK(ac::bit_cast<uint64_t>(a) == ac::bit_cast<uint64_t>(a));
    ERRCHK(ac::bit_cast<uint64_t>(a) != ac::bit_cast<uint64_t>(1.234567891011121));
    ERRCHK(ac::bit_cast<uint32_t>(d) == ac::bit_cast<uint32_t>(d));
    ERRCHK(ac::bit_cast<uint32_t>(d) != ac::bit_cast<uint32_t>(2.54123146f));

    constexpr auto path{"test.txt"};

    // Test lossless
    std::ofstream os{path};
    ERRCHK(os);
    ac::fmt::push(os, a, b, c, d);
    os.close();

    double        e;
    uint64_t      f;
    int           g;
    float         h;
    std::ifstream is{path};
    ERRCHK(is);
    ac::fmt::pull(is, e, f, g, h);
    is.close();

    PRINT_DEBUG(a);
    PRINT_DEBUG(b);
    PRINT_DEBUG(c);
    PRINT_DEBUG(d);
    PRINT_DEBUG(e);
    PRINT_DEBUG(f);
    PRINT_DEBUG(g);
    PRINT_DEBUG(h);
    ERRCHK(ac::bit_cast<uint64_t>(a) == ac::bit_cast<uint64_t>(e));
    ERRCHK(ac::bit_cast<uint32_t>(d) == ac::bit_cast<uint32_t>(h));

    return EXIT_SUCCESS;
}
