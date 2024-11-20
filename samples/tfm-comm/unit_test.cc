#include "array.h"
#include "buffer.h"
#include "buffer_exchange.h"
#include "decomp.h"
#include "math_utils.h"
#include "ndarray.h"
#include "ndvector.h"
#include "pack.h"
#include "partition.h"
#include "type_conversion.h"
#include "vector.h"

int
main(void)
{
    test_type_conversion();
    test_array();
    test_math_utils();
    test_buffer();
    test_ndarray();
    test_partition();
    test_decomp();
    test_pack();
    test_buffer_exchange();
    test_vector();
    test_ndvector();

    return EXIT_SUCCESS;
}
