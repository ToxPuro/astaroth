#include <stdio.h>
#include <stdlib.h>

#include "dynarr.h"
#include "math_utils.h"
#include "mpi_utils.h"
#include "pack.h"
#include "partition.h"
#include "print.h"
#include "type_conversion.h"

int
main(void)
{
    test_dynarr();
    test_math_utils();
    test_mpi_utils();
    test_pack();
    test_partition();
    return EXIT_SUCCESS;
}
