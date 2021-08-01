#pragma once

#include <cuComplex.h>    // CUDA complex types
#include <float.h>        // DBL/FLT_EPSILON
#include <vector_types.h> // CUDA vector types

#if AC_DOUBLE_PRECISION
typedef double AcReal;
typedef double3 AcReal3;
typedef cuDoubleComplex acComplex;
#define acComplex(x, y) make_cuDoubleComplex(x, y)
#define AC_REAL_EPSILON (DBL_EPSILON)
#define AC_MPI_TYPE (MPI_DOUBLE)
#else
typedef float AcReal;
typedef float3 AcReal3;
typedef cuFloatComplex acComplex;
#define acComplex(x, y) make_cuFloatComplex(x, y)
#define AC_REAL_EPSILON (FLT_EPSILON)
#define AC_MPI_TYPE (MPI_FLOAT)
#endif

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;