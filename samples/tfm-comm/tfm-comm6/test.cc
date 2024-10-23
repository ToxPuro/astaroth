#include "ndarray.h"
#include "static_array.h"
#include "type_conversion.h"

#include <stdlib.h>

int
main(void)
{
    test_type_conversion();
    test_static_array();
    test_ndarray();

    return EXIT_SUCCESS;
}
