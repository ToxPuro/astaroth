#include <stdio.h>
#include <stdlib.h>

#include "hip.h"
#include <hip/hip_runtime.h>

#include "errchk_gpu.h"

typedef struct {
    double* data;
    size_t count;
    bool on_device;
} AcBuffer;

AcBuffer
acBufferCreate(const size_t count, const bool on_device)
{
    AcBuffer buffer    = {.count = count, .on_device = on_device};
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
    ERRCHK_GPU_API(cudaMemcpy(out->data, in.data, sizeof(in.data[0]) * in.count, kind));
}

#include <iostream>

void
acBufferPrint(const char* label, const AcBuffer buffer)
{
    std::cout << label;

    const size_t max_print_elements = 5;
    if (buffer.on_device) {
        AcBuffer tmp = acBufferCreate(buffer.count, false);
        acBufferMigrate(buffer, &tmp);
        acBufferDestroy(&tmp);
    }
}

__global__ void
kernel(const size_t count, const double* in, double* out)
{
    const size_t i = threadIdx.x * blockIdx.x * blockDim.x;
    if (i < count)
        out[i] = 2 * in[i];
}

__global__ void
init(const size_t count, double* arr)
{
    const size_t i = threadIdx.x * blockIdx.x * blockDim.x;
    if (i < count)
        arr[i] = 1;
}

int
main(void)
{
    const size_t count = 10;
    double *in, *out;
    ERRCHK_GPU_API(cudaMalloc(&in, sizeof(in[0]) * count));
    ERRCHK_GPU_API(cudaMalloc(&out, sizeof(out[0]) * count));

    const size_t tpb = 256;
    const size_t bpg = 1;
    ERRCHK_GPU_KERNEL();

    ERRCHK_GPU_API(cudaFree(in));
    ERRCHK_GPU_API(cudaFree(out));
    return EXIT_SUCCESS;
}
