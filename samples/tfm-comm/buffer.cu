#include "buffer.h"

#if defined(__CUDACC__)
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#define __HOST_CODE_ONLY__
#include <string.h>
#endif

#include <iostream>

#include "errchk_gpu.h"

AcBuffer
acBufferCreate(const size_t count, const bool on_device)
{
    AcBuffer buffer = {
        .on_device = on_device,
        .count     = count,
        .data      = NULL,
    };
    const size_t bytes = sizeof(buffer.data[0]) * count;
    if (buffer.on_device) {
#if defined(__HOST_CODE_ONLY__)
        ERROR("Compiled in host-only mode. Cannot allocate buffers on device.");
#else
        ERRCHK_GPU_API(cudaMalloc((void**)&buffer.data, bytes));
#endif
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
#if defined(__HOST_CODE_ONLY__)
    free(buffer->data);
#else
    if (buffer->on_device)
        cudaFree(buffer->data);
    else
        free(buffer->data);
#endif
    buffer->data  = NULL;
    buffer->count = 0;
}

void
acBufferMigrate(const AcBuffer in, AcBuffer* out)
{
    ERRCHK(out->count >= in.count);
    const size_t bytes = sizeof(in.data[0]) * in.count;
#if defined(__HOST_CODE_ONLY__)
    memmove(out->data, in.data, bytes);
#else
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

    if (kind == cudaMemcpyHostToHost)
        memmove(out->data, in.data, bytes);
    else
        ERRCHK_GPU_API(cudaMemcpy(out->data, in.data, sizeof(in.data[0]) * in.count, kind));
#endif
}

void
acBufferPrint(const char* label, const AcBuffer buffer)
{
    std::cout << label << ": ";
    for (size_t i = 0; i < buffer.count; ++i)
        std::cout << buffer.data[i] << ((i + 1 < buffer.count) ? ", " : "\n");
}
