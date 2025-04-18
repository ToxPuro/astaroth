#include <cstdlib>

#include "acm/detail/math_utils.h"
#include "benchmarks/bm.h"

static int
test_median()
{
    ERRCHK(within_machine_epsilon(bm::median(std::vector<uint64_t>{1, 2, 3}), 2.));
    ERRCHK(within_machine_epsilon(bm::median(std::vector<int>{1, 2, 3, 4}), 2.5));
    ERRCHK(within_machine_epsilon(bm::median(std::vector<int>{1, 2, 3, 4, 5}), 3.));
    ERRCHK(within_machine_epsilon(bm::median(std::vector<int>{1, 2, 3, 4, 5, 6}), 3.5));

    return 0;
}

int
main()
{
    ERRCHK(test_median() == 0);
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
