#include <stdio.h>
#include <stdlib.h>

// #include "array.h"
// #include "comm.h"
// #include "matrix.h"
#include "math_utils.h"
#include "mpi_utils.h"
#include "print.h"
#include "type_conversion.h"

int
main(void)
{
    test_math_utils();
    test_mpi_utils();
    return EXIT_SUCCESS;
}
