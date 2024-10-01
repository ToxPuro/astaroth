#include <stdio.h>
#include <stdlib.h>

// #include "array.h"
// #include "comm.h"
#include "math_utils.h"
// #include "matrix.h"
#include "mpi_utils.h"
// #include "ndarray.h"
#include "print.h"
#include "type_conversion.h"

int
main(void)
{
    // test_array();
    // test_ndarray();
    test_math_utils();
    // test_matrix();
    test_mpi_utils();

    print_size_t("Test", 1);
    return EXIT_SUCCESS;
}
