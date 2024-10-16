#include "buf.h"

#include "nalloc.h"

Buffer
buffer_create(const size_t count)
{
    Buffer buffer;
    buffer.count = count;
    ncalloc(count, buffer.data);
    return buffer;
}

void
buffer_destroy(Buffer* buffer)
{
    ndealloc(buffer->data);
    buffer->count = 0;
}
