#include "buffer.h"

#include "alloc.h"
#include "print.h"

Buffer
buffer_create(const size_t count)
{
    Buffer buffer = (Buffer){
        .count = count,
        .data  = ac_calloc(count, sizeof(buffer.data[0])),
    };
    return buffer;
}

void
buffer_destroy(Buffer* buffer)
{
    ac_free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}
