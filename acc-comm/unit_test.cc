#include "array.h"
#include "buffer.h"
#include "buffer_exchange.h"
#include "decomp.h"
#include "math_utils.h"
#include "ndbuffer.h"
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
    // test_partition();
    // test_decomp();
    // test_pack();
    // test_buffer_exchange();
    // test_buffer();
    // test_ndbuffer();
    // test_vector();

    return EXIT_SUCCESS;
}
