/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
#include <functional>
#endif

#if AC_RUNTIME_COMPILATION
  #include <dlfcn.h>
#endif

#include "acc_runtime.h"
#include "user_built-in_constants.h"
#include "user_builtin_non_scalar_constants.h"

#if AC_MPI_ENABLED
#include <mpi.h>
struct AcCommunicator
{
	MPI_Comm handle;
};

typedef struct AcSubCommunicators {
	MPI_Comm x;
	MPI_Comm y;
	MPI_Comm z;

	MPI_Comm xy;
	MPI_Comm xz;
	MPI_Comm yz;
} AcSubCommunicators;
#endif

typedef enum AcBoundary {
    BOUNDARY_NONE  = 0,
    BOUNDARY_X_TOP = 0x01,
    BOUNDARY_X_BOT = 0x02,
    BOUNDARY_X     = BOUNDARY_X_TOP | BOUNDARY_X_BOT,
    BOUNDARY_Y_TOP = 0x04,
    BOUNDARY_Y_BOT = 0x08,
    BOUNDARY_Y     = BOUNDARY_Y_TOP | BOUNDARY_Y_BOT,
    BOUNDARY_Z_TOP = 0x10,
    BOUNDARY_Z_BOT = 0x20,
    BOUNDARY_Z     = BOUNDARY_Z_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_XY    = BOUNDARY_X | BOUNDARY_Y,
    BOUNDARY_XZ    = BOUNDARY_X | BOUNDARY_Z,
    BOUNDARY_YZ    = BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_XYZ   = BOUNDARY_X | BOUNDARY_Y | BOUNDARY_Z
} AcBoundary;


#define UNUSED __attribute__((unused)) // Does not give a warning if unused



typedef struct {
    AcReal* vertex_buffer[NUM_ALL_FIELDS];
    AcReal* profile[NUM_PROFILES];
    AcMeshInfo info;
} AcMesh;

typedef enum {
	STREAM_0,
	STREAM_1,
	STREAM_2,
	STREAM_3,
	STREAM_4,
	STREAM_5,
	STREAM_6,
	STREAM_7,
	STREAM_8,
	STREAM_9,
	STREAM_10,
	STREAM_11,
	STREAM_12,
	STREAM_13,
	STREAM_14,
	STREAM_15,
	STREAM_16,
	STREAM_17,
	STREAM_18,
	STREAM_19,
	STREAM_20,
	STREAM_21,
	STREAM_22,
	STREAM_23,
	STREAM_24,
	STREAM_25,
	STREAM_26,
	STREAM_27,
	STREAM_28,
	STREAM_29,
	STREAM_30,
	STREAM_31,
	STREAM_ALL,
} Stream;

const Stream STREAM_DEFAULT = STREAM_0;
#define NUM_STREAMS (32)


// For plate buffers.
enum {AC_H2D, AC_D2H};    // pack/unpack direction
typedef enum {AC_XZ=0, AC_YZ=1, AC_BOT=0, AC_TOP=2, NUM_PLATE_BUFFERS=4} PlateType;


#define RTYPE_ISNAN (RTYPE_SUM)

#define AC_FOR_INIT_TYPES(FUNC)                                                   \
    FUNC(INIT_TYPE_RANDOM)                                                        \
    FUNC(INIT_TYPE_AA_RANDOM)                                                     \
    FUNC(INIT_TYPE_XWAVE)                                                         \
    FUNC(INIT_TYPE_GAUSSIAN_RADIAL_EXPL)                                          \
    FUNC(INIT_TYPE_ABC_FLOW)                                                      \
    FUNC(INIT_TYPE_SIMPLE_CORE)                                                   \
    FUNC(INIT_TYPE_KICKBALL)                                                      \
    FUNC(INIT_TYPE_VEDGE)                                                         \
    FUNC(INIT_TYPE_VEDGEX)                                                        \
    FUNC(INIT_TYPE_RAYLEIGH_TAYLOR)                                               \
    FUNC(INIT_TYPE_RAYLEIGH_BENARD)

#define AC_GEN_ID(X) X,

typedef enum {
    AC_FOR_INIT_TYPES(AC_GEN_ID) //
    NUM_INIT_TYPES
} InitType;


#define  AcProcMappingStrategy_MORTON (-1)
#define  AcProcMappingStrategy_LINEAR (1)

#define  AcDecompStrategy_DEFAULT (-1)
#define  AcDecompStrategy_EXTERNAL (1)

#define  AcMPICommStrategy_DUPLICATE_MPI_COMM_WORLD (-1)
#define  AcMPICommStrategy_DUPLICATE_USER_COMM (1)

#undef AC_GEN_ID

#define AC_GEN_STR(X) #X,
static const char* initcondtype_names[] UNUSED = {AC_FOR_INIT_TYPES(AC_GEN_STR) "-end-"};


#undef AC_GEN_STR

typedef struct node_s* Node;
typedef struct device_s* Device;


typedef struct {
    Volume m;
    Volume n;
} GridDims;

typedef struct {
    int num_devices;
    Device* devices;

    GridDims grid;
    GridDims subgrid;
} DeviceConfiguration;





#if AC_RUNTIME_COMPILATION

#ifndef BASE_FUNC_NAME

#if __cplusplus
#define BASE_FUNC_NAME(func_name) func_name##_BASE
#else
#define BASE_FUNC_NAME(func_name) func_name
#endif

#endif

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) static UNUSED return_type (*func_name) __VA_ARGS__ = (return_type (*) __VA_ARGS__ ) ac_library_not_yet_loaded
#endif

#ifndef OVERLOADED_FUNC_DEFINE
#define OVERLOADED_FUNC_DEFINE(return_type, func_name, ...) static UNUSED return_type (*BASE_FUNC_NAME(func_name)) __VA_ARGS__ = (return_type (*) __VA_ARGS__ ) ac_library_not_yet_loaded
#endif


#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) static UNUSED return_type func_name __VA_ARGS__
#endif

#ifndef OVERLOADED_FUNC_DEFINE
#define OVERLOADED_FUNC_DEFINE FUNC_DEFINE
#endif

#ifndef BASE_FUNC_NAME 
#define BASE_FUNC_NAME(func_name) func_name
#endif

#endif

#ifdef __cplusplus

template <typename T1, typename T2, typename T3>
static inline AcReal3 
acConstructReal3Param(const T1 a, const T2 b, const T3 c, const AcMeshInfo info)
{
	return (AcReal3)
	{
		info[a],
		info[b],
		info[c]
	};
}
template <typename T1, typename T2, typename T3>
static inline int3
acConstructInt3Param(const T1 a, const T2 b, const T3 c,
                     const AcMeshInfo info)
{
    return (int3){
        info[a],
        info[b],
        info[c],
    };
}
#endif
