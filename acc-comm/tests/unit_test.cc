#include "acm/detail/allocator.h"
#include "acm/detail/buffer.h"
#include "acm/detail/buffer_exchange.h"
#include "acm/detail/decomp.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/pointer.h"
#include "acm/detail/transform.h"
#include "acm/detail/type_conversion.h"

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
    test_allocator();
    test_pointer();
    test_ntuple();
    test_buffer();
    test_ndbuffer();
    test_datatypes();

    // APIs
    test_mpi_utils();

    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
