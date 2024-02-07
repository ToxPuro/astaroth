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
    size_t bytes;
    real* data;
    bool on_device;
} Array;

Array arrayCreate(const size_t length, const bool on_device);

void arrayDestroy(Array* a);

/**
    Simple rng for reals in range [0...1].
    Not suitable for generating full-precision f64 randoms.
*/
real randd(void);

void arrayRandomize(Array* a);