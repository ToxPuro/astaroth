#include "math_utils.h"

#include "errchk.h"

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res{1};
    for (size_t i{0}; i < count; ++i)
        res *= arr[i];
    return res;
}

void
test_math_utils(void)
{
    PRINT_LOG_WARNING("Not implemented");
    // test_to_linear();
    // test_to_spatial();
    // test_within_box();
    // PRINT_LOG_INFO("OK");
}
