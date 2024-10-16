#include "buffer.h"

#include <cstring>
#include <iostream>

#include "errchk.h"

AcBuffer
acBufferCreate(const size_t count, const bool on_device)
{
    WARNCHKK(!on_device, "Requested to create the buffer on device but was unable because AcBuffer "
                         "module is compiled in host-only mode");

    AcBuffer buffer = {
        .on_device = on_device,
        .count     = count,
        .data      = (double*)malloc(sizeof(buffer.data[0]) * count),
    };
    ERRCHK(buffer.data != NULL);
    return buffer;
}

void
acBufferDestroy(AcBuffer* buffer)
{
    free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}

void
acBufferMigrate(const AcBuffer in, AcBuffer* out)
{
    ERRCHK(out->count >= in.count);
    memmove(out->data, in.data, sizeof(in.data[0]) * in.count);
}

void
acBufferPrint(const char* label, const AcBuffer buffer)
{
    std::cout << label << ": ";
    for (size_t i = 0; i < buffer.count; ++i)
        std::cout << buffer.data[i] << ((i + 1 < buffer.count) ? ", " : "\n");
}
