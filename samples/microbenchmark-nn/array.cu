#include "array.h"

#include "errchk.h"

Array
arrayCreate(const size_t length, const bool on_device)
{
    Array a = (Array){
        .length    = length,
        .bytes     = length * sizeof(real),
        .data      = NULL,
        .on_device = on_device,
    };
    if (on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&a.data, a.bytes));
    }
    else {
        a.data = (real*)malloc(a.bytes);
        ERRCHK_ALWAYS(a.data);
    }

    return a;
}

void
arrayDestroy(Array* a)
{
    if (a->on_device)
        cudaFree(a->data);
    else
        free(a->data);
    a->data   = NULL;
    a->length = 0;
}

/**
    Simple rng for reals in range [0...1].
    Not suitable for generating full-precision f64 randoms.
*/
real
randd(void)
{
    return (real)rand() / RAND_MAX;
}

void
arrayRandomize(Array* a)
{
    if (!a->on_device) {
        for (size_t i = 0; i < a->length; ++i)
            a->data[i] = randd();
    }
    else {
        Array b = arrayCreate(a->length, false);
        arrayRandomize(&b);
        cudaMemcpy(a->data, b.data, b.bytes, cudaMemcpyHostToDevice);
        arrayDestroy(&b);
    }
}