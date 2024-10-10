#include "buffer.h"

#include <iostream>

#include "errchk_gpu.h"

AcBuffer
acBufferCreate(const size_t count, const bool on_device)
{
    AcBuffer buffer    = {.on_device = on_device, .count = count, .data = NULL,};
    const size_t bytes = sizeof(buffer.data[0]) * count;
    if (buffer.on_device) {
        ERRCHK_GPU_API(cudaMalloc((void**)&buffer.data, bytes));
    }
    else {
        buffer.data = (double*)malloc(bytes);
    }
    ERRCHK(buffer.data != NULL);
    return buffer;
}

void
acBufferDestroy(AcBuffer* buffer)
{
    if (buffer->on_device)
        cudaFree(buffer->data);
    else
        free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}

void
acBufferMigrate(const AcBuffer in, AcBuffer* out)
{
    cudaMemcpyKind kind;
    if (in.on_device) {
        if (out->on_device)
            kind = cudaMemcpyDeviceToDevice;
        else
            kind = cudaMemcpyDeviceToHost;
    }
    else {
        if (out->on_device)
            kind = cudaMemcpyHostToDevice;
        else
            kind = cudaMemcpyHostToHost;
    }

    ERRCHK(out->count >= in.count);
    const size_t bytes = sizeof(in.data[0]) * in.count;
    if (kind == cudaMemcpyHostToHost)
        memmove(out->data, in.data, bytes);
    else
        ERRCHK_GPU_API(cudaMemcpy(out->data, in.data, sizeof(in.data[0]) * in.count, kind));
}

void acBufferPrint(const char* label, const AcBuffer buffer)
{
    std::cout << label << ": ";
    for (size_t i=0; i < buffer.count; ++i)
        std::cout << buffer.data[i] << ((i+1 < buffer.count) ? ", " : "\n");
}