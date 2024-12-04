#include "acm/detail/array.h"
#include "acm/detail/buffer.h"
#include "acm/detail/buffer_exchange.h"
#include "acm/detail/decomp.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/memory_resource.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/type_conversion.h"
#include "acm/detail/vector.h"

int
main(void)
{
    test_type_conversion();
    test_array();
    test_math_utils();
    test_partition();
    test_decomp();
    test_pack();
    test_buffer_exchange();
    test_buffer();
    test_ndbuffer();
    test_vector();
    test_memory_resource();

    return EXIT_SUCCESS;
}
