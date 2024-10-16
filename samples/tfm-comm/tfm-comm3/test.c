#include <stdio.h>
#include <stdlib.h>

#include "comm.h"

int
main(void)
{
    int errcount = 0;
    errcount += acCommTest();
    if (errcount == 0)
        printf("---C test success---\n");
    else
        printf("---C test failed: %d errors found---\n", errcount);
    return errcount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
