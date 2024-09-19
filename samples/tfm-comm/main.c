#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"
#include "ndarray.h"

int
main(void)
{
    comm_run();
    test_math_utils();
    ndarray_test();

    return EXIT_SUCCESS;
}
