#include <cstdlib>

#include "acm/detail/errchk.h"
#include "acm/detail/math_utils.h"

static int
test_intersect_lines(void)
{
    ERRCHK(intersect_lines(0, 1, 1, 2) == false);
    ERRCHK(intersect_lines(0, 3, 1, 1) == true);
    ERRCHK(intersect_lines(0, 3, 2, 3) == true);
    ERRCHK(intersect_lines(1, 2, 0, 1) == false);
    ERRCHK(intersect_lines(1, 2, 0, 2) == true);
    ERRCHK(intersect_lines(1, 3, 0, 4) == true);
    ERRCHK(intersect_lines(0, 4, 1, 3) == true);

    return 0;
}

int
main()
{
    ERRCHK(test_intersect_lines() == 0);
    return EXIT_SUCCESS;
}
