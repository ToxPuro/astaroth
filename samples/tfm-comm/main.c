#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"

int
main(void)
{
    test_math_utils();
    comm_run();
    return EXIT_SUCCESS;
}
