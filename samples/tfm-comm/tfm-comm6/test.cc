#include "comm.h"
#include "ndarray.h"
#include "partition.h"
#include "static_array.h"
#include "type_conversion.h"

#include <stdlib.h>

int
main(void)
{
    test_type_conversion();
    test_static_array();
    test_ndarray();
    test_partition();
    acCommTest();

    return EXIT_SUCCESS;
}
