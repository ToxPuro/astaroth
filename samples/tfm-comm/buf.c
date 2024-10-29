#include "buf.h"

#include "errchk.h"
// #include "nalloc.h"

Buffer
buffer_create(const size_t count)
{
    Buffer buffer = {
        .count = count,
        .data  = calloc(count, sizeof(buffer.data[0])),
    };
    ERRCHK(buffer.data);
    return buffer;
}

void
buffer_destroy(Buffer* buffer)
{
    free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}
