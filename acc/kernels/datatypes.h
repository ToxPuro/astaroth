#pragma once

#include <cuComplex.h>    // CUDA complex types
#include <vector_types.h> // CUDA vector types

#if AC_DOUBLE_PRECISION == 1
typedef double AcReal;
typedef double3 AcReal3;
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#else
typedef float AcReal;
typedef float3 AcReal3;
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#endif

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;