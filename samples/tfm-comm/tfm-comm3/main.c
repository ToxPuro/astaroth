#include <stdio.h>
#include <stdlib.h>

#include "alloc.h"
#include "buffer.h"
#include "print.h"
#include "shape.h"

int
main(void)
{
    printf("Hello\n");
    const size_t count = 50;
    Buffer buf0        = bufferCreate(count);
    PRINT_BUFFER(buf0);
    bufferDestroy(&buf0);

    const size_t ndims = 3;
    Shape shape        = shapeCreate(ndims);
    PRINT_SHAPE(shape);
    shapeDestroy(&shape);

    return EXIT_SUCCESS;
}
