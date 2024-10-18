#include "print.h"
#include "static_array.h"
#include "type_conversion.h"

#include "errchk.h"

int
main(void)
{
    int retval = 0;

    retval |= test_type_conversion();
    retval |= test_static_array();

    return retval;
}
