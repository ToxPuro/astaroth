#include "buffer.h"

#include "alloc.h"
#include "print.h"

Buffer
bufferCreate(const size_t count)
{
    Buffer buffer = (Buffer){
        .count = count,
        .data  = ac_calloc(count, sizeof(buffer.data[0])),
    };
    return buffer;
}

void
bufferDestroy(Buffer* buffer)
{
    ac_free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}
