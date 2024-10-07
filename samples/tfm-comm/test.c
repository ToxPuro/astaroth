#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "dynarr.h"
#include "math_utils.h"
#include "mpi_utils.h"
#include "pack.h"
#include "partition.h"
// #include "print.h"
#include "segment.h"
#include "type_conversion.h"

int
main(void)
{
    test_comm();
    test_dynarr();
    test_math_utils();
    test_mpi_utils();
    test_pack();
    test_partition();
    test_segment();
    return EXIT_SUCCESS;
}
