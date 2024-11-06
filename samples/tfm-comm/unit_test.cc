#include "buffer.h"
#include "buffer_exchange.h"
#include "decomp.h"
#include "math_utils.h"
#include "ndarray.h"
#include "partition.h"
#include "static_array.h"
#include "type_conversion.h"

int
main(void)
{
    test_type_conversion();
    test_static_array();
    test_ndarray();
    test_partition();
    test_decomp();
    test_buffer();
    test_math_utils();
    test_buffer_exchange();

    return EXIT_SUCCESS;
}
