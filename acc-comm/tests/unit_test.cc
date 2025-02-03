#include "acm/detail/array.h"
#include "acm/detail/buffer.h"
#include "acm/detail/buffer_exchange.h"
#include "acm/detail/decomp.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/memory_resource.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/pointer.h"
#include "acm/detail/static_array.h"
#include "acm/detail/transform.h"
#include "acm/detail/type_conversion.h"
#include "acm/detail/vector.h"

int
main(void)
{
    // Operations
    test_type_conversion();
    test_math_utils();
    test_partition();
    test_decomp();
    test_pack();
    test_buffer_exchange();
    test_transform();

    // Data types
    test_array();
    test_buffer();
    test_ndbuffer();
    test_vector();
    test_static_array();
    test_memory_resource();
    test_pointer();

    // APIs
    test_mpi_utils();

    return EXIT_SUCCESS;
}
