#pragma once

#include "acreal.h"
#include <stddef.h>
#ifndef VOLUME_DEFINED

typedef struct
{ 
	size_t x,y,z;
} Volume;
#define VOLUME_DEFINED
#endif

#ifndef COMPLEX_DEFINED

typedef struct
{ 
	AcReal x,y;
} AcComplex;
#define COMPLEX_DEFINED
#endif

#ifndef REAL3_DEFINED

typedef struct
{ 
	AcReal x,y,z;
} AcReal3;
#define REAL3_DEFINED
#endif

typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1, AC_NOT_ALLOCATED = 2} AcResult;

#define N_DIMS (3)
#define X_ORDER_INT (0)
#define Y_ORDER_INT (1)
#define Z_ORDER_INT (2)

typedef enum {
	XYZ = X_ORDER_INT + N_DIMS*Y_ORDER_INT + N_DIMS*N_DIMS*Z_ORDER_INT,
	XZY = X_ORDER_INT + N_DIMS*Z_ORDER_INT + N_DIMS*N_DIMS*Y_ORDER_INT,
	YXZ = Y_ORDER_INT + N_DIMS*X_ORDER_INT + N_DIMS*N_DIMS*Z_ORDER_INT,
	YZX = Y_ORDER_INT + N_DIMS*Z_ORDER_INT + N_DIMS*N_DIMS*X_ORDER_INT,
	ZXY = Z_ORDER_INT + N_DIMS*X_ORDER_INT + N_DIMS*N_DIMS*Y_ORDER_INT,
	ZYX = Z_ORDER_INT + N_DIMS*Y_ORDER_INT + N_DIMS*N_DIMS*X_ORDER_INT,
} AcMeshOrder;

typedef enum AcReduceOp
{
	NO_REDUCE,
	REDUCE_MIN,
	REDUCE_MAX,
	REDUCE_SUM,
} AcReduceOp;

#define ONE_DIMENSIONAL_PROFILE (1 << 20)
#define TWO_DIMENSIONAL_PROFILE (1 << 21)
typedef enum {
	PROFILE_NONE = 0,
	PROFILE_X  = (1 << 0) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Y  = (1 << 1) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_Z  = (1 << 2) | ONE_DIMENSIONAL_PROFILE,
	PROFILE_XY = (1 << 3) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_XZ = (1 << 4) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YX = (1 << 5) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_YZ = (1 << 6) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZX = (1 << 7) | TWO_DIMENSIONAL_PROFILE,
	PROFILE_ZY = (1 << 8) | TWO_DIMENSIONAL_PROFILE,
} AcProfileType;

typedef struct {
  size_t x, y, z, w;
} AcShape;

typedef Volume size3_t;

//TP: opaque pointer for the MPI comm to enable having the opaque type in modules which do not about MPI_Comm
typedef struct AcCommunicator AcCommunicator;

typedef struct {
    size3_t n0, n1;
    size3_t m0, m1;
    size3_t nn;
    size3_t reduction_tile;
} AcMeshDims;

#if AC_CPU_BUILD

#ifndef INT3_DEFINED
typedef struct
{
	int x,y,z;
} int3;
#define INT3_DEFINED
#endif

typedef struct
{
    unsigned int x, y, z;
} dim3;

typedef struct
{
    unsigned int x, y, z;
} uint3;
#endif
