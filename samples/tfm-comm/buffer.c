#include "buffer.h"

#include <stdlib.h>

#include "errchk.h"
#include "math_utils.h"
#include "print.h"

Buffer
acBufferCreate(const size_t length)
{
    Buffer buf = (Buffer){
        .length = length,
        .data   = calloc(length, sizeof(buf.data[0])),
    };
    ERRCHK(buf.data);
    return buf;
}

void
acBufferPrint(const char* label, const Buffer buffer)
{
    printf("Buffer %s:\n", label);
    print("\tlength", buffer.length);

    const size_t max_print_elements = 5;
    print_array("\tdata (max first 5 elems)", min(buffer.length, max_print_elements), buffer.data);
}

void
acBufferDestroy(Buffer* buf)
{
    free(buf->data);
    buf->length = 0;
}
