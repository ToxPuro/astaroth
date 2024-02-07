#pragma once
#include "errchk.h"

#if AC_DOUBLE_PRECISION
#define DOUBLE_PRECISION (1)
typedef double real;
#else
#define DOUBLE_PRECISION (0)
typedef float real;
#endif

typedef struct {
    size_t length;
    real* data;
    bool on_device;
} Array;

static Array
arrayCreate(const size_t length, const bool on_device)
{
    Array a = (Array){
        .length    = length,
        .data      = NULL,
        .on_device = on_device,
    };

    const size_t bytes = length * sizeof(a.data[0]);
    if (on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&a.data, bytes));
    }
    else {
        a.data = (real*)malloc(bytes);
        ERRCHK_ALWAYS(a.data);
    }

    return a;
}

static void
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
static real
randd(void)
{
    return (real)rand() / RAND_MAX;
}

static void
arrayRandomize(Array* a)
{
    if (!a->on_device) {
        for (size_t i = 0; i < a->length; ++i)
            a->data[i] = randd();
    }
    else {
        Array b = arrayCreate(a->length, false);
        arrayRandomize(&b);
        const size_t bytes = a->length * sizeof(b.data[0]);
        cudaMemcpy(a->data, b.data, bytes, cudaMemcpyHostToDevice);
        arrayDestroy(&b);
    }
}