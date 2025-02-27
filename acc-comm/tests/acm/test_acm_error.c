#include <stdio.h>
#include <stdlib.h>

#include "acm/acm_error.h"
#include "acm/detail/errchk_print.h"

#define ERRCHK(expr)                                                                               \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            errchk_print_error(__func__, __FILE__, __LINE__, #expr, "");                           \
            errchk_print_stacktrace();                                                             \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

int
main(void)
{
    ERRCHK(ACM_Test_get_errorcode_description() == 0);
    return EXIT_SUCCESS;
}
