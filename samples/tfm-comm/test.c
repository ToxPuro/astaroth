#include "test.h"

#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "print.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

int
main(void)
{
    printf("Testing...\n");
    test_math_utils();
    printf("Complete\n");
    return EXIT_SUCCESS;
}
