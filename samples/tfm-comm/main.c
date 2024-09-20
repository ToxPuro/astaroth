#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "math_utils.h"
#include "ndarray.h"

int
main(void)
{
    // Init
    acCommInit();

    // Compute
    acCommRun();
    test_math_utils();
    ndarray_test();

    // Quit
    acCommQuit();

    return EXIT_SUCCESS;
}
