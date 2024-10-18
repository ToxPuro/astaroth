#include "print.h"
#include "static_array.h"
#include "type_conversion.h"

#include <stdexcept>

#include "errchk.h"
// #define ERRCHK_THROW(expr)                                                                         \
//     (ERRCHK(expr) == 0 ? 0 : (throw std::runtime_error("ERRCHK failure"), -1))

int
main(void)
{
    test_type_conversion();

    return EXIT_SUCCESS;
}
