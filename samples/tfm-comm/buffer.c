#include "buffer.h"

#include <stdlib.h>

#include "errchk.h"

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
acBufferDestroy(Buffer* buf)
{
    free(buf->data);
    buf->length = 0;
}
