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

#if AC_MPI_ENABLED
#include <mpi.h>
#endif


#define _UNUSED __attribute__((unused)) // Does not give a warning if unused

  typedef struct AcMeshInfo{
  AcMeshInfoParams params;
#if AC_MPI_ENABLED
    MPI_Comm comm;
#endif

#ifdef __cplusplus
#include "info_access_operators.h"
#endif
    AcCompInfo run_consts;
  } AcMeshInfo;


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

#ifdef __cplusplus
enum class AcProcMappingStrategy:int{
    Morton = -1,//The default
    Linear  = 1,
    Hierarchical= 2, 
};
enum class AcDecomposeStrategy:int{
    Morton = -1, 
    External = 1,
    Hierarchical= 2, 
};
enum class AcMPICommStrategy:int{
	DuplicateMPICommWorld = -1,
	DuplicateUserComm = 1,
};
#endif

#define  AcProcMappingStrategy_MORTON (-1)
#define  AcProcMappingStrategy_LINEAR (1)

#define  AcDecompStrategy_DEFAULT (-1)
#define  AcDecompStrategy_EXTERNAL (1)

#define  AcMPICommStrategy_DUPLICATE_MPI_COMM_WORLD (-1)
#define  AcMPICommStrategy_DUPLICATE_USER_COMM (1)

#undef AC_GEN_ID

#define AC_GEN_STR(X) #X,
static const char* initcondtype_names[] _UNUSED = {AC_FOR_INIT_TYPES(AC_GEN_STR) "-end-"};


#undef AC_GEN_STR
#undef _UNUSED

typedef struct node_s* Node;
typedef struct device_s* Device;


typedef struct {
    int3 m;
    int3 n;
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
#define FUNC_DEFINE(return_type, func_name, ...) static UNUSED return_type (*func_name) __VA_ARGS__
#endif

#ifndef OVERLOADED_FUNC_DEFINE
#define OVERLOADED_FUNC_DEFINE(return_type, func_name, ...) static return_type (*BASE_FUNC_NAME(func_name)) __VA_ARGS__
#endif


#else

#ifndef FUNC_DEFINE
#define FUNC_DEFINE(return_type, func_name, ...) return_type func_name __VA_ARGS__
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
/** Sets the dimensions of the computational domain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */

extern "C" {
#endif

int3 acConstructInt3Param(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                          const AcMeshInfo info);

static inline AcReal3
acConstructReal3Param(const AcRealParam a, const AcRealParam b, const AcRealParam c,
                     const AcMeshInfo info)
{
    return (AcReal3){
        info.params.scalars.real_params[a],
        info.params.scalars.real_params[b],
        info.params.scalars.real_params[c],
    };
}

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */

FUNC_DEFINE(Volume, acGetLocalNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetLocalMM, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridMM, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetMinNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(Volume, acGetGridMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(AcReal3, acGetLengths, (const AcMeshInfo info));


static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);
    return mm.x*mm.y*mm.z;
}
static inline size_t
acGridVertexBufferSize(const AcMeshInfo info)
{
    const Volume mm = acGetGridMM(info);
    return mm.x*mm.y*mm.z;
}

static inline Volume 
acVertexBufferDims(const AcMeshInfo info)
{
    return acGetLocalMM(info);
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferSize(info);
}


static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info)
{
    const Volume nn = acGetLocalNN(info);
    return nn.x*nn.y*nn.z;
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
   const Volume n0 = acGetMinNN(info);
   const Volume n1 = acGetMaxNN(info);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = acGetLocalMM(info);
   const Volume nn = acGetLocalNN(info);
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

static inline AcMeshDims
acGetGridMeshDims(const AcMeshInfo info)
{
   const Volume n0 = acGetMinNN(info);
   const Volume n1 = acGetGridMaxNN(info);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = acGetGridMM(info);
   const Volume nn = acGetGridNN(info);
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.params.scalars.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

FUNC_DEFINE(size_t, acGetKernelId,(const AcKernel kernel));


FUNC_DEFINE(AcResult, acAnalysisGetKernelInfo,(const AcMeshInfoParams info, KernelAnalysisInfo* dst));
	



FUNC_DEFINE(size_t, acGetKernelIdByName,(const char* name));


FUNC_DEFINE(AcMeshInfo, acGridDecomposeMeshInfo,(const AcMeshInfo global_config));

#if AC_RUNTIME_COMPILATION == 0
FUNC_DEFINE(VertexBufferArray, acGridGetVBA,(void));
#endif

FUNC_DEFINE(AcMeshInfo, acGridGetLocalMeshInfo,(void));

#ifdef __cplusplus
//TP: this is done for perf optim since if acVertexBufferIdx is called often
//Making it an external function call is quite expensive
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    #include "user_builtin_non_scalar_constants.h"
    auto mm = info[AC_mgrid];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    #include "user_builtin_non_scalar_constants.h"
    auto mm = info[AC_mlocal];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#else
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const Volume mm = acGetGridMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#endif

static inline int3
acVertexBufferSpatialIdx(const size_t i, const AcMeshInfo info)
{
    const Volume mm = acGetLocalMM(info);

    return (int3){
        (int)i % (int)mm.x,
        ((int)i % (int)(mm.x * mm.y)) / (int)mm.x,
        (int)i / (int)(mm.x * mm.y),
    };
}

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
    {
        printf("[%s]: %d\n", intparam_names[i], config.params.scalars.int_params[i]);
    }
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
    {
        printf("[%s]: (%d, %d, %d)\n", int3param_names[i],config.params.scalars.int3_params[i].x, config.params.scalars.int3_params[i].y, config.params.scalars.int3_params[i].z);
    }
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
    {
        printf("[%s]: %g\n", realparam_names[i], (double)(config.params.scalars.real_params[i]));
    }
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
    {
        printf("[%s]: (%g, %g, %g)\n", real3param_names[i], (double)(config.params.scalars.real3_params[i].x),
							    (double)(config.params.scalars.real3_params[i].y),
							    (double)(config.params.scalars.real3_params[i].z)
	      );
    }
}


/** Prints a list of initial condition condition types */
static inline void
acQueryInitcondtypes(void)
{
    for (int i = 0; i < NUM_INIT_TYPES; ++i)
        printf("%s (%d)\n", initcondtype_names[i], i);
}


/** Prints a list of int parameters */
static inline void
acQueryIntparams(void)
{
    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        printf("%s (%d)\n", intparam_names[i], i);
}

/** Prints a list of int3 parameters */
static inline void
acQueryInt3params(void)
{
    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        printf("%s (%d)\n", int3param_names[i], i);
}

/** Prints a list of real parameters */
static inline void
acQueryRealparams(void)
{
    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        printf("%s (%d)\n", realparam_names[i], i);
}

/** Prints a list of real3 parameters */
static inline void
acQueryReal3params(void)
{
    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        printf("%s (%d)\n", real3param_names[i], i);
}

/** Prints a list of Scalar array handles */
/*
static inline void
acQueryScalarrays(void)
{
    for (int i = 0; i < NUM_REAL_ARRS_1D; ++i)
        printf("%s (%d)\n", realarr1D_names[i], i);
}
*/

/** Prints a list of vertex buffer handles */
static inline void
acQueryVtxbufs(void)
{
    for (int i = 0; i < NUM_ALL_FIELDS; ++i)
        printf("%s (%d)\n", vtxbuf_names[i], i);
}

/** Prints a list of kernels */
static inline void
acQueryKernels(void)
{
    for (int i = 0; i < NUM_KERNELS; ++i)
        printf("%s (%d)\n", kernel_names[i], i);
}

static inline void
acPrintIntParam(const AcIntParam a, const AcMeshInfo info)
{
    printf("%s: %d\n", intparam_names[a], info.params.scalars.int_params[a]);
}

static inline void
acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c, const AcMeshInfo info)
{
    acPrintIntParam(a, info);
    acPrintIntParam(b, info);
    acPrintIntParam(c, info);
}

static inline void
acPrintInt3Param(const AcInt3Param a, const AcMeshInfo info)
{
    const int3 vec = info.params.scalars.int3_params[a];
    printf("{%s: (%d, %d, %d)}\n", int3param_names[a], vec.x, vec.y, vec.z);
}

/*
 * =============================================================================
 * Legacy interface
 * =============================================================================
 */

FUNC_DEFINE(AcResult, acCheckDeviceAvailability,(void));
FUNC_DEFINE(int, acGetNumDevicesPerNode,(void));

/** Returns the number of fields (vertexbuffer handles). */
FUNC_DEFINE(size_t, acGetNumFields,(void));

/** Gets the field handle corresponding to a null-terminated `str` and stores the result in
 * `handle`.
 *
 * Returns AC_SUCCESS on success.
 * Returns AC_FAILURE if the field was not found and sets `handle` to SIZE_MAX.
 *
 * Example usage:
 * ```C
 * size_t handle;
 * AcResult res = acGetFieldHandle("VTXBUF_LNRHO", &handle);
 * if (res != AC_SUCCESS)
 *  fprintf(stderr, "Handle not found\n");
 * ```
 *  */
FUNC_DEFINE(AcResult, acGetFieldHandle,(const char* field, size_t* handle));

/** */
FUNC_DEFINE(Node, acGetNode,(void));

/*
 * =============================================================================
 * Grid interface
 * =============================================================================
 */
#if AC_MPI_ENABLED

/**
Calls MPI_Init and creates a separate communicator for Astaroth procs with MPI_Comm_split, color =
666 Any program running in the same MPI process space must also call MPI_Comm_split with some color
!= 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI and creating a communicator.

Returns AC_FAILURE otherwise.
 */
FUNC_DEFINE(AcResult, ac_MPI_Init,(void));

/**
Calls MPI_Init_thread with the provided thread_level and creates a separate communicator for
Astaroth procs with MPI_Comm_split, color = 666 Any program running in the same MPI process space
must also call MPI_Comm_split with some color != 666. OTHERWISE this call will hang.

Returns AC_SUCCESS on successfullly initializing MPI with the requested thread level and creating a
communicator.

Returns AC_FAILURE otherwise.
 */
FUNC_DEFINE(AcResult, ac_MPI_Init_thread,(int thread_level));

/**
Destroys the communicator and calls MPI_Finalize
*/
FUNC_DEFINE(void, ac_MPI_Finalize,());

/** Returns the rank of the Astaroth communicator */
int ac_MPI_Comm_rank();

/** If MPI was initialized with MPI_Init* instead of ac_MPI_Init, this will return MPI_COMM_WORLD */
FUNC_DEFINE(MPI_Comm, acGridMPIComm,());
/** Returns the size of the Astaroth communicator */
FUNC_DEFINE(int, ac_MPI_Comm_size,());

/** Calls MPI_Barrier on the Astaroth communicator */
FUNC_DEFINE(void, ac_MPI_Barrier,());

/**
Initializes all available devices.

Must compile and run the code with MPI.

Must allocate exactly one process per GPU. And the same number of processes
per node as there are GPUs on that node.

Devices in the grid are configured based on the contents of AcMesh.
 */

FUNC_DEFINE(AcResult, acGridInitBase, (const AcMesh mesh));
static AcResult UNUSED 
acGridInit(const AcMeshInfo info)
{
	AcMesh mesh;
	for(int i = 0; i < NUM_ALL_FIELDS; ++i) mesh.vertex_buffer[i] = NULL;
	mesh.info = info;
	return acGridInitBase(mesh);
}

/**
Resets all devices on the current grid.
 */
FUNC_DEFINE(AcResult, acGridQuit,(void));

/** Get the local device */
FUNC_DEFINE(Device, acGridGetDevice,(void));

/** Randomizes the local mesh */
FUNC_DEFINE(AcResult, acGridRandomize,(void));

/** */
FUNC_DEFINE(AcResult, acGridSynchronizeStream,(const Stream stream));

/** */
FUNC_DEFINE(AcResult, acGridLoadScalarUniform,(const Stream stream, const AcRealParam param, const AcReal value));

/** */
FUNC_DEFINE(AcResult, acGridLoadVectorUniform,(const Stream stream, const AcReal3Param param,
                                 const AcReal3 value));

/** */
FUNC_DEFINE(AcResult, acGridLoadIntUniform,(const Stream stream, const AcIntParam param, const int value));

/** */
FUNC_DEFINE(AcResult, acGridLoadInt3Uniform,(const Stream stream, const AcInt3Param param, const int3 value));

/** */
FUNC_DEFINE(AcResult, acGridLoadMesh,(const Stream stream, const AcMesh host_mesh));

/** */
FUNC_DEFINE(AcResult, acGridStoreMesh,(const Stream stream, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acGridIntegrate,(const Stream stream, const AcReal dt));

FUNC_DEFINE(AcResult, acGridSwapBuffers,(void));

/** */
/*   MV: Commented out for a while, but save for the future when standalone_MPI
         works with periodic boundary conditions.
AcResult
acGridIntegrateNonperiodic(const Stream stream, const AcReal dt)

AcResult acGridIntegrateNonperiodic(const Stream stream, const AcReal dt);
*/

/** */
FUNC_DEFINE(AcResult, acGridPeriodicBoundconds,(const Stream stream));


/** */
FUNC_DEFINE(AcResult, acGridReduceScal,(const Stream stream, const AcReduction reduction,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acGridReduceVec,(const Stream stream, const AcReduction reduction,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acGridReduceVecScal,(const Stream stream, const AcReduction reduction,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result));

/** */
AcResult acGridReduceXY(const Stream stream, const Field field, const Profile profile, const AcReduction reduction);

typedef enum {
    ACCESS_READ,
    ACCESS_WRITE,
} AccessType;

FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronous,(const VertexBufferHandle field, const char* dir,
                                           const char* label, const AccessType type));

FUNC_DEFINE(AcResult, acGridDiskAccessLaunch,(const AccessType type));

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
FUNC_DEFINE(AcResult, acGridWriteSlicesToDiskLaunch,(const char* dir, const char* label));

/* Synchronous */
FUNC_DEFINE(AcResult, acGridWriteSlicesToDiskCollectiveSynchronous,(const char* dir, const char* label));

/* Asynchronous. Need to call acGridDiskAccessSync afterwards */
FUNC_DEFINE(AcResult, acGridWriteMeshToDiskLaunch,(const char* dir, const char* label));

FUNC_DEFINE(AcResult, acGridDiskAccessSync,(void));

FUNC_DEFINE(AcResult, acGridReadVarfileToMesh,(const char* file, const Field fields[], const size_t num_fields,
                                 const int3 nn, const int3 rr));

/* Quick hack for the hero run, will be removed in future builds */
FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronousDistributed,(const VertexBufferHandle vtxbuf,
                                                      const char* dir, const char* label,
                                                      const AccessType type));

/* Quick hack for the hero run, will be removed in future builds */
FUNC_DEFINE(AcResult, acGridAccessMeshOnDiskSynchronousCollective,(const VertexBufferHandle vtxbuf,
                                                     const char* dir, const char* label,
                                                     const AccessType type));

// Bugged
// AcResult acGridLoadFieldFromFile(const char* path, const VertexBufferHandle field);

// Bugged
// AcResult acGridStoreFieldToFile(const char* path, const VertexBufferHandle field);

/*
 * =============================================================================
 * Task interface (part of the grid interface)
 * =============================================================================
 */

/** */
typedef enum AcTaskType {
    TASKTYPE_COMPUTE,
    TASKTYPE_HALOEXCHANGE,
    TASKTYPE_BOUNDCOND,
    TASKTYPE_SYNC,
    TASKTYPE_REDUCE,
} AcTaskType;

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




FUNC_DEFINE(acAnalysisBCInfo, acAnalysisGetBCInfo,(const AcMeshInfoParams info, const AcKernel bc, const AcBoundary boundary));

typedef struct ParamLoadingInfo {
        acKernelInputParams* params;
        Device device;
        const int step_number;
        const int3 boundary_normal;
        const Field vtxbuf;
} ParamLoadingInfo;

//opaque for C to enable C++ lambdas
typedef struct LoadKernelParamsFunc LoadKernelParamsFunc;

/** TaskDefinition is a datatype containing information necessary to generate a set of tasks for
 * some operation.*/
typedef struct AcTaskDefinition {
    AcTaskType task_type;
    AcKernel kernel_enum;
    AcBoundary boundary;

    Field* fields_in;
    size_t num_fields_in;

    Field* fields_out;
    size_t num_fields_out;

    Profile* profiles_in;
    size_t num_profiles_in;

    Profile* profiles_out;
    size_t num_profiles_out;

    AcRealParam* parameters;
    size_t num_parameters;
    LoadKernelParamsFunc* load_kernel_params_func;
    bool fieldwise;

    KernelReduceOutput* outputs_in;
    size_t num_outputs_in;

    KernelReduceOutput* outputs_out;
    size_t num_outputs_out;
} AcTaskDefinition;

/** TaskGraph is an opaque datatype containing information necessary to execute a set of
 * operations.*/
typedef struct AcTaskGraph AcTaskGraph;

#if __cplusplus
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acComputeWithParams,(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out,Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out, 
			   KernelReduceOutput reduce_outputs_in[], size_t num_outputs_in, KernelReduceOutput reduce_outputs_out[], size_t num_outputs_out,
			   std::function<void(ParamLoadingInfo step_info)> loader));
#else
/** */
FUNC_DEFINE(AcTaskDefinition, acComputeWithParams,(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out,Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out, void (*load_func)(ParamLoadingInfo step_info)));
#endif

/** */
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acCompute,(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out,Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out));

#if __cplusplus
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acBoundaryCondition,
		(const AcBoundary boundary, const AcKernel kernel, const Field fields_in[], const size_t num_fields_in, const Field fields_out[], const size_t num_fields_out, const std::function<void(ParamLoadingInfo step_info)>));
#else
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acBoundaryCondition,
		(const AcBoundary boundary, AcKernel kernel, Field fields_in[], const size_t num_fields_in, Field fields_out[], const size_t num_fields_out,void (*load_func)(ParamLoadingInfo step_info)));
#endif
/** */
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acHaloExchange,(Field fields[], const size_t num_fields));

FUNC_DEFINE(AcTaskDefinition, acSync,());
/** */
FUNC_DEFINE(AcTaskGraph*, acGridGetDefaultTaskGraph,());

/** */
FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsX,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsY,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(bool, acGridTaskGraphHasPeriodicBoundcondsZ,(AcTaskGraph* graph));

/** */
OVERLOADED_FUNC_DEFINE(AcTaskGraph*, acGridBuildTaskGraph,(const AcTaskDefinition ops[], const size_t n_ops));

/** */
FUNC_DEFINE(AcTaskGraph*, acGetDSLTaskGraph,(const AcDSLTaskGraph));
FUNC_DEFINE(AcTaskGraph*, acGetOptimizedDSLTaskGraph,(const AcDSLTaskGraph));


/** */
FUNC_DEFINE(AcResult, acGridDestroyTaskGraph,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(AcResult, acGridExecuteTaskGraph,(AcTaskGraph* graph, const size_t n_iterations));
/** */
FUNC_DEFINE(AcResult, acGridExecuteTaskGraphBase,(AcTaskGraph* graph, const size_t n_iterations, const bool include_all));
/** */
FUNC_DEFINE(AcResult, acGridFinalizeReduceLocal,(AcTaskGraph* graph));
/** */
FUNC_DEFINE(AcResult, acGridFinalizeReduce,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(AcResult, acGridLaunchKernel,(const Stream stream, const AcKernel kernel, const Volume start,
                            const Volume end));


/** */
FUNC_DEFINE(AcResult, acGridLoadStencil,(const Stream stream, const Stencil stencil,
                           const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

/** */
FUNC_DEFINE(AcResult, acGridStoreStencil,(const Stream stream, const Stencil stencil,
                            AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

/** */
FUNC_DEFINE(AcResult, acGridLoadStencils,(const Stream stream,
                   const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

/** */
FUNC_DEFINE(AcResult, acGridStoreStencils,(const Stream stream,
                    AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

#endif // AC_MPI_ENABLED

/*
 * =============================================================================
 * Node interface
 * =============================================================================
 */
/**
Initializes all devices on the current node.

Devices on the node are configured based on the contents of AcMesh.

@return Exit status. Places the newly created handle in the output parameter.
@see AcMeshInfo


Usage example:
@code
AcMeshInfo info;
acLoadConfig(AC_DEFAULT_CONFIG, &info);

Node node;
acNodeCreate(0, info, &node);
acNodeDestroy(node);
@endcode
 */
FUNC_DEFINE(AcResult, acNodeCreate,(const int id, const AcMeshInfo node_config, Node* node));

/**
Resets all devices on the current node.

@see acNodeCreate()
 */
FUNC_DEFINE(AcResult, acNodeDestroy,(Node node));

/**
Prints information about the devices available on the current node.

Requires that Node has been initialized with
@See acNodeCreate().
*/
FUNC_DEFINE(AcResult, acNodePrintInfo,(const Node node));

/**



@see DeviceConfiguration
*/
FUNC_DEFINE(AcResult, acNodeQueryDeviceConfiguration,(const Node node, DeviceConfiguration* config));

/** */
FUNC_DEFINE(AcResult, acNodeAutoOptimize,(const Node node));

/** */
FUNC_DEFINE(AcResult, acNodeSynchronizeStream,(const Node node, const Stream stream));

/** Deprecated ? */
FUNC_DEFINE(AcResult, acNodeSynchronizeVertexBuffer,(const Node node, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle)); // Not in Device

/** */
FUNC_DEFINE(AcResult, acNodeSynchronizeMesh,(const Node node, const Stream stream)); // Not in Device

/** */
FUNC_DEFINE(AcResult, acNodeSwapBuffers,(const Node node));

/** */
FUNC_DEFINE(AcResult, acNodeLoadConstant,(const Node node, const Stream stream, const AcRealParam param,
                            const AcReal value));

/** Deprecated ? Might be useful though if the user wants to load only one vtxbuf. But in this case
 * the user should supply a AcReal* instead of vtxbuf_handle */
FUNC_DEFINE(AcResult, acNodeLoadVertexBufferWithOffset,(const Node node, const Stream stream,
                                          const AcMesh host_mesh,
                                          const VertexBufferHandle vtxbuf_handle, const int3 src,
                                          const int3 dst, const int num_vertices));

/** */
FUNC_DEFINE(AcResult, acNodeLoadMeshWithOffset,(const Node node, const Stream stream, const AcMesh host_mesh,
                                  const int3 src, const int3 dst, const int num_vertices));

/** Deprecated ? */
FUNC_DEFINE(AcResult, acNodeLoadVertexBuffer,(const Node node, const Stream stream, const AcMesh host_mesh,
                                const VertexBufferHandle vtxbuf_handle));

/** */
FUNC_DEFINE(AcResult, acNodeLoadMesh,(const Node node, const Stream stream, const AcMesh host_mesh));

/** */
FUNC_DEFINE(AcResult, acNodeSetVertexBuffer,(const Node node, const Stream stream,
                               const VertexBufferHandle handle, const AcReal value));

/** Deprecated ? */
FUNC_DEFINE(AcResult, acNodeStoreVertexBufferWithOffset,(const Node node, const Stream stream,
                                           const VertexBufferHandle vtxbuf_handle, const int3 src,
                                           const int3 dst, const int num_vertices,
                                           AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acNodeStoreMeshWithOffset,(const Node node, const Stream stream, const int3 src,
                                   const int3 dst, const int num_vertices, AcMesh* host_mesh));

/** Deprecated ? */
FUNC_DEFINE(AcResult, acNodeStoreVertexBuffer,(const Node node, const Stream stream,
                                 const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acNodeStoreMesh,(const Node node, const Stream stream, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acNodeIntegrateSubstep,(const Node node, const Stream stream, const int step_number,
                                const int3 start, const int3 end, const AcReal dt));

/** */
FUNC_DEFINE(AcResult, acNodeIntegrate,(const Node node, const AcReal dt));

/** */
FUNC_DEFINE(AcResult, acNodeIntegrateGBC,(const Node node, const AcMeshInfo config, const AcReal dt));

/** */
FUNC_DEFINE(AcResult, acNodePeriodicBoundcondStep,(const Node node, const Stream stream,
                                     const VertexBufferHandle vtxbuf_handle));

/** */
FUNC_DEFINE(AcResult, acNodePeriodicBoundconds,(const Node node, const Stream stream));

/** */
FUNC_DEFINE(AcResult, acNodeGeneralBoundcondStep,(const Node node, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle,
                                    const AcMeshInfo config));

/** */
FUNC_DEFINE(AcResult, acNodeGeneralBoundconds,(const Node node, const Stream stream, const AcMeshInfo config));

/** */
FUNC_DEFINE(AcResult, acNodeReduceScal,(const Node node, const Stream stream, const AcReduction reduction,
                          const VertexBufferHandle vtxbuf_handle, AcReal* result));
/** */
FUNC_DEFINE(AcResult, acNodeReduceVec,(const Node node, const Stream stream_type, const AcReduction reduction,
                         const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                         const VertexBufferHandle vtxbuf2, AcReal* result));
/** */
FUNC_DEFINE(AcResult, acNodeReduceVecScal,(const Node node, const Stream stream_type, const AcReduction reduction,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                             AcReal* result));
/** */
FUNC_DEFINE(AcResult, acNodeLoadPlate,(const Node node, const Stream stream, const int3 start, const int3 end, 
                         AcMesh* host_mesh, AcReal* plateBuffer, int plate));
/** */
FUNC_DEFINE(AcResult, acNodeStorePlate,(const Node node, const Stream stream, const int3 start, const int3 end,
                          AcMesh* host_mesh, AcReal* plateBuffer, int plate));
/** */
FUNC_DEFINE(AcResult, acNodeStoreIXYPlate,(const Node node, const Stream stream, const int3 start, const int3 end, 
                             AcMesh* host_mesh, int plate));
/** */
FUNC_DEFINE(AcResult, acNodeLoadPlateXcomp,(const Node node, const Stream stream, const int3 start, const int3 end, 
                              AcMesh* host_mesh, AcReal* plateBuffer, int plate));

#if AC_RUNTIME_COMPILATION == 0
/** */
FUNC_DEFINE(AcResult, acNodeGetVBApointers,(Node* node_handle, AcReal *vbapointer[2]));
#endif

/*
 * =============================================================================
 * Device interface
 * =============================================================================
 */
/** */
FUNC_DEFINE(AcResult, acDeviceCreate,(const int id, const AcMeshInfo device_config, Device* device));

/** */
FUNC_DEFINE(AcResult, acDeviceDestroy,(Device* device));

/** Resets the mesh to default values defined in acc_runtime.cu:acVBAReset */
FUNC_DEFINE(AcResult, acDeviceResetMesh,(const Device device, const Stream stream));

/** */
FUNC_DEFINE(AcResult, acDevicePrintInfo,(const Device device));

/** */
// AcResult acDeviceAutoOptimize(const Device device);

/** */
FUNC_DEFINE(AcResult, acDeviceSynchronizeStream,(const Device device, const Stream stream));

/** */
FUNC_DEFINE(AcResult, acDeviceSwapBuffer,(const Device device, const VertexBufferHandle handle));

/** */
FUNC_DEFINE(AcResult, acDeviceSwapBuffers,(const Device device));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadScalarUniform,(const Device device, const Stream stream,
                                   const AcRealParam param, const AcReal value));
FUNC_DEFINE(AcResult, acDevicePrintProfiles,(const Device device));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadVectorUniform,(const Device device, const Stream stream,
                                   const AcReal3Param param, const AcReal3 value));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadIntUniform,(const Device device, const Stream stream, const AcIntParam param,
                                const int value));
/** */
FUNC_DEFINE(AcResult, acDeviceLoadBoolUniform,(const Device device, const Stream stream, const AcBoolParam param,
                                const bool value));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadInt3Uniform,(const Device device, const Stream stream, const AcInt3Param param,
                                 const int3 value));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreScalarUniform,(const Device device, const Stream stream,
                                    const AcRealParam param, AcReal* value));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreVectorUniform,(const Device device, const Stream stream,
                                    const AcReal3Param param, AcReal3* value));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadMeshInfo,(const Device device, const AcMeshInfo device_config));


/** */
FUNC_DEFINE(AcResult, acDeviceLoadVertexBufferWithOffset,(const Device device, const Stream stream,
                                            const AcMesh host_mesh,
                                            const VertexBufferHandle vtxbuf_handle, const int3 src,
                                            const int3 dst, const int num_vertices));

/** Deprecated */
FUNC_DEFINE(AcResult, acDeviceLoadMeshWithOffset,(const Device device, const Stream stream,
                                    const AcMesh host_mesh, const int3 src, const int3 dst,
                                    const int num_vertices));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadVertexBuffer,(const Device device, const Stream stream, const AcMesh host_mesh,
                                  const VertexBufferHandle vtxbuf_handle));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadMesh,(const Device device, const Stream stream, const AcMesh host_mesh));

/** */

#define DEVICE_LOAD_ARRAY_DECL(ENUM,DEF_NAME) \
	FUNC_DEFINE(AcResult, acDeviceLoad##DEF_NAME##Array,(const Device device, const Stream stream, const AcMeshInfo host_info, const ENUM array));

#include "device_load_uniform_decl.h"
#define DECL_DEVICE_STORE_UNIFORM(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	FUNC_DEFINE(AcResult,acDeviceStore##VAL_TYPE_UPPER_CASE##Uniform,(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value));
#define DECL_DEVICE_STORE_ARRAY(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	FUNC_DEFINE(AcResult,acDeviceStore##VAL_TYPE_UPPER_CASE##Array,(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value));
#include "device_store_uniform_decl.h"


/** */
FUNC_DEFINE(AcResult, acDeviceSetVertexBuffer,(const Device device, const Stream stream,
                                 const VertexBufferHandle handle, const AcReal value));

/** */
FUNC_DEFINE(AcResult, acDeviceFlushOutputBuffers,(const Device device, const Stream stream));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreVertexBufferWithOffset,(const Device device, const Stream stream,
                                             const VertexBufferHandle vtxbuf_handle, const int3 src,
                                             const int3 dst, const int num_vertices,
                                             AcMesh* host_mesh));
FUNC_DEFINE(AcMeshInfo, acDeviceGetConfig,(const Device device));

FUNC_DEFINE(acKernelInputParams*, acDeviceGetKernelInputParamsObject,(const Device device));


/** Deprecated */
FUNC_DEFINE(AcResult, acDeviceStoreMeshWithOffset,(const Device device, const Stream stream, const int3 src,
                                     const int3 dst, const int num_vertices, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreVertexBuffer,(const Device device, const Stream stream,
                                   const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreMesh,(const Device device, const Stream stream, AcMesh* host_mesh));


/** */
FUNC_DEFINE(AcResult, acDeviceTransferVertexBufferWithOffset,(const Device src_device, const Stream stream,
                                                const VertexBufferHandle vtxbuf_handle,
                                                const int3 src, const int3 dst,
                                                const int num_vertices, Device dst_device));

/** Deprecated */
FUNC_DEFINE(AcResult, acDeviceTransferMeshWithOffset,(const Device src_device, const Stream stream,
                                        const int3 src, const int3 dst, const int num_vertices,
                                        Device* dst_device));

/** */
FUNC_DEFINE(AcResult, acDeviceTransferVertexBuffer,(const Device src_device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, Device dst_device));

/** */
FUNC_DEFINE(AcResult, acDeviceTransferMesh,(const Device src_device, const Stream stream, Device dst_device));

/** */
FUNC_DEFINE(AcResult, acDeviceIntegrateSubstep,(const Device device, const Stream stream, const int step_number,
                                  const Volume start, const Volume end, const AcReal dt));
/** */
FUNC_DEFINE(AcResult, acDevicePeriodicBoundcondStep,(const Device device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 start,
                                       const int3 end));

/** */
FUNC_DEFINE(AcResult, acDevicePeriodicBoundconds,(const Device device, const Stream stream, const int3 start,
                                    const int3 end));

/** */
FUNC_DEFINE(AcResult, acDeviceGeneralBoundcondStep,(const Device device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, const int3 start,
                                      const int3 end, const AcMeshInfo config, const int3 bindex));

/** */
FUNC_DEFINE(AcResult, acDeviceGeneralBoundconds,(const Device device, const Stream stream, const int3 start,
                                   const int3 end, const AcMeshInfo config, const int3 bindex));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceScalNoPostProcessing,(const Device device, const Stream stream,
                                       const AcReduction reduction,
                                       const VertexBufferHandle vtxbuf_handle, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceScal,(const Device device, const Stream stream, const AcReduction reduction,
                            const VertexBufferHandle vtxbuf_handle, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceVecNoPostProcessing,(const Device device, const Stream stream_type,
                                      const AcReduction reduction, const VertexBufferHandle vtxbuf0,
                                      const VertexBufferHandle vtxbuf1,
                                      const VertexBufferHandle vtxbuf2, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceVec,(const Device device, const Stream stream_type, const AcReduction reduction,
                           const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                           const VertexBufferHandle vtxbuf2, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceVecScalNoPostProcessing,(const Device device, const Stream stream_type,
                                          const AcReduction reduction,
                                          const VertexBufferHandle vtxbuf0,
                                          const VertexBufferHandle vtxbuf1,
                                          const VertexBufferHandle vtxbuf2,
                                          const VertexBufferHandle vtxbuf3, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceVecScal,(const Device device, const Stream stream_type,
                               const AcReduction reduction, const VertexBufferHandle vtxbuf0,
                               const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2,
                               const VertexBufferHandle vtxbuf3, AcReal* result));

/** */
FUNC_DEFINE(AcResult, acDeviceReduceXY,(const Device device, const Stream stream, const Field field,
                                 const Profile profile, const AcReduction reduction));

/** */
FUNC_DEFINE(AcResult, acDeviceSwapProfileBuffer,(const Device device, const Profile handle));
/** */
FUNC_DEFINE(AcResult, acDeviceReduceAverages,(const Device device, const Stream stream, const Profile prof));
/** */
FUNC_DEFINE(AcBuffer, acDeviceTransposeBase,(const Device device, const Stream stream, const AcMeshOrder order, const AcReal* src));
/** */
static UNUSED AcBuffer
acDeviceTranspose(const Device device, const Stream stream, const AcMeshOrder order, const AcReal* src)
{
	return acDeviceTransposeBase(device,stream,order,src);
}
FUNC_DEFINE(AcBuffer, acDeviceTransposeVertexBuffer,(const Device device, const Stream stream, const AcMeshOrder order, const VertexBufferHandle vtxbuf));
/** */

/** */
FUNC_DEFINE(AcResult, acDeviceSwapProfileBuffers,(const Device device, const Profile* profiles,
                                    const size_t num_profiles));

/** */
FUNC_DEFINE(AcResult, acDeviceSwapAllProfileBuffers,(const Device device));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadProfile,(const Device device, const AcReal* hostprofile,
                             const size_t hostprofile_count, const Profile profile));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreProfile,(const Device device, const Profile profile, AcMesh* host_mesh));

/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceReal,(Device device, const Stream stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceFloat,(Device device, const Stream stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceInt,(Device device, const Stream stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceRealStream,(Device device, const cudaStream_t stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceFloatStream,(Device device, const cudaStream_t stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceIntStream,(Device device, const cudaStream_t stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output));
/** */
FUNC_DEFINE(AcResult,acDevicePreprocessScratchPad,(Device device, const int variable, const AcType type,const AcReduceOp op));

/** */
FUNC_DEFINE(AcResult, acDeviceUpdate,(Device device, const AcMeshInfo info));

/** */
FUNC_DEFINE(AcDeviceKernelOutput, acDeviceGetKernelOutput,(const Device device));


/** */
FUNC_DEFINE(AcResult, acDeviceLaunchKernel,(const Device device, const Stream stream, const AcKernel kernel,
                              const Volume start, const Volume end));

/** */
FUNC_DEFINE(AcResult, acDeviceBenchmarkKernel,(const Device device, const AcKernel kernel, const int3 start,
                                 const int3 end));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadStencilsFromConfig,(const Device device, const Stream stream));

FUNC_DEFINE(AcResult, acDeviceLoadStencil,(const Device device, const Stream stream, const Stencil stencil,const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));
/** */
FUNC_DEFINE(AcResult, acDeviceLoadStencils,(const Device device, const Stream stream, const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));
/** */
FUNC_DEFINE(AcResult, acDeviceStoreStencil,(const Device device, const Stream stream, const Stencil stencil,AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]));

/** */
FUNC_DEFINE(AcResult, acDeviceVolumeCopy,(const Device device, const Stream stream,const AcReal* in, const Volume in_offset, const Volume in_volume,AcReal* out, const Volume out_offset, const Volume out_volume));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadPlateBuffer,(const Device device, int3 start, int3 end, const Stream stream,
                                 AcReal* buffer, int plate));

/** */
FUNC_DEFINE(AcResult, acDeviceStorePlateBuffer,(const Device device, int3 start, int3 end, const Stream stream, 
                                  AcReal* buffer, int plate));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreIXYPlate,(const Device device, int3 start, int3 end, int src_offset, const Stream stream, 
                               AcMesh *host_mesh));

/** */
FUNC_DEFINE(AcMeshInfo,acDeviceGetLocalConfig,(const Device device));

FUNC_DEFINE(AcResult, acDeviceGetVertexBufferPtrs,(Device device, const VertexBufferHandle vtxbuf, AcReal** in, AcReal** out));

/** */
AcResult acDeviceWriteMeshToDisk(const Device device, const VertexBufferHandle vtxbuf,
                                 const char* filepath);


/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
AcResult 
acHostUpdateBuiltinParams(AcMeshInfo* config);
AcResult 
acHostUpdateBuiltinCompParams(AcCompInfo* comp_config);



FUNC_DEFINE(AcReal*, acHostCreateVertexBuffer,(const AcMeshInfo info));
FUNC_DEFINE(AcResult, acHostMeshCreateProfiles,(AcMesh* mesh));
FUNC_DEFINE(AcResult, acHostMeshDestroyVertexBuffer,(AcReal** vtxbuf));
/** Creates a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));
/** Copies the VertexBuffers from src to dst*/
FUNC_DEFINE(AcResult, acHostMeshCopyVertexBuffers,(const AcMesh src, AcMesh dst));
/** Copies a host mesh to a new host mesh */
FUNC_DEFINE(AcResult, acHostMeshCopy,(const AcMesh src, AcMesh* dst));
/** Creates a mesh stored in host memory (size of the whole grid) */
FUNC_DEFINE(AcResult, acHostGridMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));

/** Checks that the loaded dynamic Astaroth is binary compatible with the loader */
FUNC_DEFINE(AcResult, acVerifyCompatibility, (const size_t mesh_size, const size_t mesh_info_size, const size_t params_size, const size_t comp_info_size, const int num_reals, const int num_ints, const int num_bools, const int num_real_arrays, const int num_int_arrays, const int num_bool_arrays));

/** Randomizes a host mesh */
FUNC_DEFINE(AcResult, acHostMeshRandomize,(AcMesh* mesh));
/** Randomizes a host mesh (uses n[xyz]grid params)*/
FUNC_DEFINE(AcResult, acHostGridMeshRandomize,(AcMesh* mesh));

/** Destroys a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshDestroy,(AcMesh* mesh));

FUNC_DEFINE(void, acStoreConfig,(const AcMeshInfo info, const char* filename));

/** Sets the dimensions of the computational domain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */
AcResult acSetMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

/*
 * =============================================================================
 * Logging functions
 * =============================================================================
 */

/* Log a message with a timestamp from the root proc (if pid == 0) */
FUNC_DEFINE(void, acLogFromRootProc,(const int pid, const char* msg, ...));
FUNC_DEFINE(void, acVA_LogFromRootProc,(const int pid, const char* msg, va_list args));

/* Log a message with a timestamp from the root proc (if pid == 0) if the build flag VERBOSE is on
 */
FUNC_DEFINE(void, acVerboseLogFromRootProc,(const int pid, const char* msg, ...));
FUNC_DEFINE(void, acVA_VerboseLogFromRootProc,(const int pid, const char* msg, va_list args));

/* Log a message with a timestamp from the root proc (if pid == 0) in a debug build */
FUNC_DEFINE(void, acDebugFromRootProc,(const int pid, const char* msg, ...));
FUNC_DEFINE(void, acVA_DebugFromRootProc,(const int pid, const char* msg, va_list arg));


#include "device_set_input_decls.h"
#include "device_get_output_decls.h"
#include "device_get_input_decls.h"
#include "get_vtxbufs_declares.h"

#if AC_RUNTIME_COMPILATION
#include "astaroth_lib.h"

#define LOAD_DSYM(FUNC_NAME) *(void**)(&FUNC_NAME) = dlsym(handle,#FUNC_NAME); \
			     if(!FUNC_NAME) fprintf(stderr,"Astaroth error: was not able to load %s\n",#FUNC_NAME);

  static AcLibHandle __attribute__((unused)) acLoadLibrary()
  {
	acLoadRunTime();
 	void* handle = dlopen(runtime_astaroth_path,RTLD_NOW);
	if(!handle)
	{
    		fprintf(stderr,"%s","Fatal error was not able to load Astaroth\n"); 
		fprintf(stderr,"Error message: %s\n",dlerror());
		exit(EXIT_FAILURE);
	}
	LOAD_DSYM(acDeviceGetVertexBufferPtrs)
	LOAD_DSYM(acDeviceGetLocalConfig)
        LOAD_DSYM(acDeviceFinishReduceInt)
        LOAD_DSYM(acKernelFlushInt)
        LOAD_DSYM(acAnalysisGetKernelInfo)
        LOAD_DSYM(acDeviceSwapAllProfileBuffers)
#if AC_MPI_ENABLED
	LOAD_DSYM(BASE_FUNC_NAME(acBoundaryCondition))
	LOAD_DSYM(ac_MPI_Init)
	LOAD_DSYM(ac_MPI_Init_thread)
	LOAD_DSYM(ac_MPI_Finalize);
	LOAD_DSYM(acGridMPIComm);
	LOAD_DSYM(acGridDecomposeMeshInfo);
	LOAD_DSYM(acGridGetLocalMeshInfo);
	LOAD_DSYM(acGridQuit);
	LOAD_DSYM(acGridGetDevice);
	LOAD_DSYM(acGridRandomize);
	LOAD_DSYM(acGridSynchronizeStream);
	LOAD_DSYM(acGridLoadScalarUniform);
	LOAD_DSYM(acGridLoadVectorUniform);
	LOAD_DSYM(acGridLoadIntUniform);
	LOAD_DSYM(acGridLoadInt3Uniform);
	LOAD_DSYM(acGridLoadMesh);
	LOAD_DSYM(acGridStoreMesh);
	LOAD_DSYM(acGridIntegrate);
	LOAD_DSYM(acGridSwapBuffers);
	LOAD_DSYM(acGridPeriodicBoundconds);
	LOAD_DSYM(acGridReduceScal);
	LOAD_DSYM(acGridReduceVec);
	LOAD_DSYM(acGridReduceVecScal);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronous);
	LOAD_DSYM(acGridDiskAccessLaunch);
	LOAD_DSYM(acGridWriteSlicesToDiskLaunch);
	LOAD_DSYM(acGridWriteSlicesToDiskCollectiveSynchronous);
	LOAD_DSYM(acGridWriteMeshToDiskLaunch);
	LOAD_DSYM(acGridDiskAccessSync);
	LOAD_DSYM(acGridReadVarfileToMesh);
	*(void**)(&BASE_FUNC_NAME(acComputeWithParams)) = dlsym(handle,"acComputeWithParams");
	*(void**)(&BASE_FUNC_NAME(acCompute)) = dlsym(handle,"acCompute");
	*(void**)(&BASE_FUNC_NAME(acHaloExchange)) = dlsym(handle,"acHaloExchange");
	*(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle,"acGridBuildTaskGraph");
	*(void**)(&acGridDestroyTaskGraph) = dlsym(handle,"acGridDestroyTaskGraph");
	*(void**)(&(acGetDSLTaskGraph)) = dlsym(handle,"acGetDSLTaskGraph");
	if(!acGetDSLTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetDSLTaskGraph");
	if(!acGridDestroyTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridDestroyTaskGraph");
	LOAD_DSYM(acGetOptimizedDSLTaskGraph);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousDistributed);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousCollective);
	LOAD_DSYM(acGridGetDefaultTaskGraph);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsX);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsY);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsZ);
	LOAD_DSYM(acGridExecuteTaskGraph);
	LOAD_DSYM(acGridExecuteTaskGraphBase);
	LOAD_DSYM(acGridFinalizeReduceLocal);
	LOAD_DSYM(acGridFinalizeReduce);
	LOAD_DSYM(acGridLaunchKernel);
	LOAD_DSYM(acGridLoadStencil);
	LOAD_DSYM(acGridStoreStencil);
	LOAD_DSYM(acGridLoadStencils);
	LOAD_DSYM(acGridStoreStencils);
	LOAD_DSYM(acGridInitBase);
#endif
	LOAD_DSYM(acGetLocalNN)
	LOAD_DSYM(acGetLocalMM)
	LOAD_DSYM(acGetGridNN)
	LOAD_DSYM(acGetGridMM)
	LOAD_DSYM(acGetMinNN)
	LOAD_DSYM(acGetMaxNN)
	LOAD_DSYM(acGetGridMaxNN)
	LOAD_DSYM(acGetLengths)
	LOAD_DSYM(acHostMeshCopyVertexBuffers)
#include "device_load_uniform_loads.h"
	LOAD_DSYM(acHostMeshCopy)
	LOAD_DSYM(acGetKernelId)
	LOAD_DSYM(acGetKernelIdByName)
	LOAD_DSYM(acCheckDeviceAvailability)
	LOAD_DSYM(acGetNumDevicesPerNode)
	LOAD_DSYM(acGetNumFields)
	LOAD_DSYM(acGetFieldHandle)
	LOAD_DSYM(acGetNode)
	LOAD_DSYM(acNodeCreate)
	LOAD_DSYM(acNodeDestroy)
	LOAD_DSYM(acNodePrintInfo)
	LOAD_DSYM(acNodeQueryDeviceConfiguration)
	LOAD_DSYM(acNodeAutoOptimize)
	LOAD_DSYM(acNodeSynchronizeStream)
	LOAD_DSYM(acNodeSynchronizeVertexBuffer)
	LOAD_DSYM(acNodeSynchronizeMesh)
	LOAD_DSYM(acNodeSwapBuffers)
	LOAD_DSYM(acNodeLoadConstant)
	LOAD_DSYM(acNodeLoadVertexBufferWithOffset)
	LOAD_DSYM(acNodeLoadMeshWithOffset)
	LOAD_DSYM(acNodeLoadVertexBuffer)
	LOAD_DSYM(acNodeLoadMesh)
	LOAD_DSYM(acNodeSetVertexBuffer)
	LOAD_DSYM(acNodeStoreVertexBufferWithOffset)
	LOAD_DSYM(acNodeStoreMeshWithOffset)
	LOAD_DSYM(acNodeStoreVertexBuffer)
	LOAD_DSYM(acNodeStoreMesh)
	LOAD_DSYM(acNodeIntegrateSubstep)
	LOAD_DSYM(acNodeIntegrate)
	LOAD_DSYM(acNodeIntegrateGBC)
	LOAD_DSYM(acNodePeriodicBoundcondStep)
	LOAD_DSYM(acNodePeriodicBoundconds)
	LOAD_DSYM(acNodeGeneralBoundcondStep)
	LOAD_DSYM(acNodeGeneralBoundconds)
	LOAD_DSYM(acNodeReduceScal)
	LOAD_DSYM(acNodeReduceVec)
	LOAD_DSYM(acNodeReduceVecScal)
	LOAD_DSYM(acNodeLoadPlate)
	LOAD_DSYM(acNodeStorePlate)
	LOAD_DSYM(acNodeStoreIXYPlate)
	LOAD_DSYM(acNodeLoadPlateXcomp)
	LOAD_DSYM(acDeviceCreate)
	LOAD_DSYM(acDeviceDestroy)
	LOAD_DSYM(acDeviceResetMesh)
	LOAD_DSYM(acDevicePrintInfo)
	LOAD_DSYM(acDeviceSynchronizeStream)
	LOAD_DSYM(acDeviceSwapBuffer)
	LOAD_DSYM(acDeviceSwapBuffers)
	LOAD_DSYM(acDeviceLoadScalarUniform)
	LOAD_DSYM(acDeviceLoadVectorUniform)
	LOAD_DSYM(acDeviceLoadIntUniform)
	LOAD_DSYM(acDeviceLoadBoolUniform)
	LOAD_DSYM(acDeviceLoadInt3Uniform)
	LOAD_DSYM(acDeviceStoreScalarUniform)
	LOAD_DSYM(acDeviceStoreVectorUniform)
	LOAD_DSYM(acDeviceStoreIntUniform)
	LOAD_DSYM(acDeviceStoreBoolUniform)
	LOAD_DSYM(acDeviceStoreInt3Uniform)
	LOAD_DSYM(acDeviceLoadMeshInfo)
	LOAD_DSYM(acDeviceLoadVertexBufferWithOffset)
	LOAD_DSYM(acDeviceLoadMeshWithOffset)
	LOAD_DSYM(acDeviceLoadVertexBuffer)
	LOAD_DSYM(acDeviceLoadMesh)
	LOAD_DSYM(acDeviceSetVertexBuffer)
	LOAD_DSYM(acDeviceFlushOutputBuffers)
	LOAD_DSYM(acDeviceStoreVertexBufferWithOffset)
	LOAD_DSYM(acDeviceGetConfig)
	LOAD_DSYM(acDeviceGetKernelInputParamsObject)
	LOAD_DSYM(acDeviceStoreMeshWithOffset)
	LOAD_DSYM(acDeviceStoreVertexBuffer)
	LOAD_DSYM(acDeviceStoreMesh)
	LOAD_DSYM(acDeviceTransferVertexBufferWithOffset)
	LOAD_DSYM(acDeviceTransferMeshWithOffset)
	LOAD_DSYM(acDeviceTransferVertexBuffer)
	LOAD_DSYM(acDeviceTransferMesh)
	LOAD_DSYM(acDeviceIntegrateSubstep)
	LOAD_DSYM(acDevicePeriodicBoundcondStep)
	LOAD_DSYM(acDevicePeriodicBoundconds)
	LOAD_DSYM(acDeviceGeneralBoundcondStep)
	LOAD_DSYM(acDeviceGeneralBoundconds)
	LOAD_DSYM(acDeviceReduceScalNoPostProcessing)
	LOAD_DSYM(acDeviceReduceScal)
	LOAD_DSYM(acDeviceReduceVecNoPostProcessing)
	LOAD_DSYM(acDeviceReduceVec)
	LOAD_DSYM(acDeviceReduceVecScalNoPostProcessing)
	LOAD_DSYM(acDeviceReduceVecScal)
	LOAD_DSYM(acDeviceUpdate)
	LOAD_DSYM(acDeviceGetKernelOutput)
	LOAD_DSYM(acDeviceLaunchKernel)
	LOAD_DSYM(acDeviceBenchmarkKernel)
	LOAD_DSYM(acDeviceLoadStencil)
	LOAD_DSYM(acDeviceLoadStencils)
	LOAD_DSYM(acDeviceLoadStencilsFromConfig)
	LOAD_DSYM(acDeviceStoreStencil)
	LOAD_DSYM(acDeviceVolumeCopy)
	LOAD_DSYM(acDeviceLoadPlateBuffer)
	LOAD_DSYM(acDeviceStorePlateBuffer)
	LOAD_DSYM(acDeviceStoreIXYPlate)
#include "device_set_input_loads.h"
#include "device_get_input_loads.h"
#include "device_get_output_loads.h"
#include "get_vtxbufs_loads.h"

	*(void**)(&acDeviceGetIntOutput) = dlsym(handle,"acDeviceGetIntOutput");
	*(void**)(&acDeviceGetRealInput) = dlsym(handle,"acDeviceGetRealInput");
	*(void**)(&acDeviceGetIntInput) = dlsym(handle,"acDeviceGetIntInput");
	*(void**)(&acDeviceGetRealOutput) = dlsym(handle,"acDeviceGetRealOutput");
	LOAD_DSYM(acHostMeshCreate)
	LOAD_DSYM(acHostGridMeshCreate)
	LOAD_DSYM(acHostMeshRandomize);
	LOAD_DSYM(acHostGridMeshRandomize);
	LOAD_DSYM(acHostMeshDestroy);
	LOAD_DSYM(acLogFromRootProc);
	LOAD_DSYM(acVA_LogFromRootProc);
	LOAD_DSYM(acVerboseLogFromRootProc);
	LOAD_DSYM(acVA_VerboseLogFromRootProc);
	LOAD_DSYM(acDebugFromRootProc);
	LOAD_DSYM(acVA_DebugFromRootProc);
	LOAD_DSYM(acVerifyCompatibility);
	LOAD_DSYM(acStoreConfig);
//#ifdef __cplusplus
//	return AcLibHandle(handle);
//#else
//	return handle;
//#endif
	const AcResult is_compatible = acVerifyCompatibility(sizeof(AcMesh), sizeof(AcMeshInfo), sizeof(AcMeshInfoParams), sizeof(AcCompInfo), NUM_REAL_PARAMS, NUM_INT_PARAMS, NUM_BOOL_PARAMS, NUM_REAL_ARRAYS, NUM_INT_ARRAYS, NUM_BOOL_ARRAYS);
	if(is_compatible == AC_FAILURE)
	{
		fprintf(stderr,"Library is not compatible\n");
		exit(EXIT_FAILURE);
	}
	return handle;
  }
#endif

/** Inits the profile to cosine wave */
AcResult acHostInitProfileToCosineWave(const long double box_size, const size_t nz,
                                       const long offset, const AcReal amplitude,
                                       const AcReal wavenumber, const size_t profile_count,
                                       AcReal* profile);

/** Inits the profile to sine wave */
AcResult acHostInitProfileToSineWave(const long double box_size, const size_t nz, const long offset,
                                     const AcReal amplitude, const AcReal wavenumber,
                                     const size_t profile_count, AcReal* profile);

/** Initialize a profile to a constant value */
AcResult acHostInitProfileToValue(const long double value, const size_t profile_count,
                                  AcReal* profile);

/** Writes the host profile to a file */
AcResult acHostWriteProfileToFile(const char* filepath, const AcReal* profile,
                                  const size_t profile_count);

/*
 * =============================================================================
 * AcBuffer
 * =============================================================================
 */

AcBuffer acBufferCreate(const AcShape shape, const bool on_device);
AcBuffer acTransposeBuffer(const AcBuffer src, const AcMeshOrder order, const cudaStream_t stream);

AcShape  acGetTransposeBufferShape(const AcMeshOrder order, const Volume dims);
AcShape  acGetReductionShape(const AcProfileType type, const AcMeshDims dims);
AcResult acReduceProfile(const Profile prof, const AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream);

AcBuffer
acBufferRemoveHalos(const AcBuffer buffer_in, const int3 halo_sizes, const cudaStream_t stream);

void acBufferDestroy(AcBuffer* buffer);

AcResult acBufferMigrate(const AcBuffer in, AcBuffer* out);
AcBuffer acBufferCopy(const AcBuffer in, const bool on_device);

#ifdef __cplusplus
} // extern "C"
#endif


#ifdef __cplusplus

static UNUSED AcResult
acDeviceFinishReduce(Device device, const Stream stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output)
{
	return acDeviceFinishReduceInt(device,stream,result,kernel,reduce_op,output);
}

static UNUSED AcResult
acDeviceFinishReduce(Device device, const Stream stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output)
{
	return acDeviceFinishReduceReal(device,stream,result,kernel,reduce_op,output);
}

static UNUSED AcResult
acDeviceFinishReduce(Device device, const Stream stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output)
{
	return acDeviceFinishReduceFloat(device,stream,result,kernel,reduce_op,output);
}

static UNUSED AcResult
acDeviceFinishReduce(Device device, const cudaStream_t stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output)
{
	return acDeviceFinishReduceIntStream(device,stream,result,kernel,reduce_op,output);
}

static UNUSED AcResult
acDeviceFinishReduce(Device device, const cudaStream_t stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output)
{
	return acDeviceFinishReduceRealStream(device,stream,result,kernel,reduce_op,output);
}

static UNUSED AcResult
acDeviceFinishReduce(Device device, const cudaStream_t stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output)
{
	return acDeviceFinishReduceFloatStream(device,stream,result,kernel,reduce_op,output);
}

#if AC_MPI_ENABLED
static UNUSED AcResult
acGridInit(const AcMesh mesh)
{
	return acGridInitBase(mesh);
}
static UNUSED AcBuffer
acDeviceTranspose(const Device device, const Stream stream, const AcMeshOrder order, const VertexBufferHandle vtxbuf)
{
	return acDeviceTransposeVertexBuffer(device,stream,order,vtxbuf);
}
#endif


#define OVERLOAD_DEVICE_STORE_UNIFORM(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	static UNUSED AcResult acDeviceStore(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value) { return acDeviceStore##VAL_TYPE_UPPER_CASE##Uniform(device,stream,param,value); }
#define OVERLOAD_DEVICE_STORE_ARRAY(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	static UNUSED AcResult acDeviceStore(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value) { return acDeviceStore##VAL_TYPE_UPPER_CASE##Array(device,stream,param,value); }
#define OVERLOAD_DEVICE_LOAD_ARRAY(ENUM,DEF_NAME) \
	static UNUSED AcResult acDeviceLoad(const Device device, const Stream stream, const AcMeshInfo host_info, const ENUM array) { return acDeviceLoad##DEF_NAME##Array(device,stream,host_info,array);}

#include "device_store_overloads.h"
#include "device_load_uniform_overloads.h"
#include "device_set_input_overloads.h"
#include "device_get_input_overloads.h"
#include "device_get_output_overloads.h"


#if AC_MPI_ENABLED
/** Backwards compatible interface, input fields = output fields*/
template <size_t num_fields>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields)[num_fields])
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields, num_fields, fields, num_fields, NULL, 0, NULL, 0);
}
static __attribute__((unused)) AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields)
{
	return BASE_FUNC_NAME(acCompute)(kernel, fields.data(), fields.size(), fields.data(), fields.size(), NULL, 0, NULL, 0);
}

/** */
template <size_t num_fields>
static AcTaskDefinition
acComputeWithParams(AcKernel kernel, Field (&fields)[num_fields], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields, num_fields, fields, num_fields, NULL, 0, NULL, 0, loader);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(aBoundaryCondition)(boundary, kernel, fields_in, num_fields_in, fields_out, num_fields_out, loader);
}

template <size_t num_fields>
static AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, Field (&fields)[num_fields], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields, num_fields, fields, num_fields, loader);
}


template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out])
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields_in, num_fields_in, fields_out, num_fields_out);
}

template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acComputeWithParams(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in, num_fields_in, fields_out, num_fields_out, loader);
}

static inline AcTaskDefinition
acComputeWithParams(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0 , NULL, 0, NULL, 0, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0, NULL, 0, NULL, 0, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::vector<Profile> profile_in, std::vector<Profile> profile_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), profile_in.data(), profile_in.size(), profile_out.data(), profile_out.size(), 
		    			       NULL, 0, NULL, 0,
		                               loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::vector<Profile> profile_in, std::vector<Profile> profile_out, std::vector<KernelReduceOutput> reduce_outputs_in, std::vector<KernelReduceOutput> reduce_outputs_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), profile_in.data(), profile_in.size(), profile_out.data(), profile_out.size(), 
		    			       reduce_outputs_in.data(), reduce_outputs_in.size(), reduce_outputs_out.data(), reduce_outputs_out.size(),
		    	                       loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out)
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields.data(), fields.size(), fields.data(), fields.size(), NULL, 0 , NULL, 0, NULL, 0, NULL, 0, loader);
}


template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in, num_fields_in, fields_out, num_fields_out, NULL, 0, NULL, 0, NULL, 0, NULL, 0, loader);
}

/** */
template <size_t num_fields>
static AcTaskDefinition
acHaloExchange(Field (&fields)[num_fields])
{
    return BASE_FUNC_NAME(acHaloExchange)(fields, num_fields);
}

AcTaskDefinition
static inline acHaloExchange(std::vector<Field> fields)
{
    return BASE_FUNC_NAME(acHaloExchange)(fields.data(), fields.size());
}

static inline
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(), fields.data(), fields.size(), loader);
}
template <typename T>
static inline
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields, const T param)
{
   
    auto loader = 
    [&](ParamLoadingInfo p)
    {
            auto config = acDeviceGetLocalConfig(p.device);
	    acLoadKernelParams(*p.params,kernel,p.vtxbuf,config[param]); 
    };
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(), fields.data(), fields.size(), loader);
}

static inline
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p){(void)p;};
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), loader);
}
static inline
AcTaskDefinition
acBoundaryCondition(const AcBoundary boundary, AcKernel kernel, std::vector<Field> fields)
{
    std::function<void(ParamLoadingInfo)> loader = [](const ParamLoadingInfo& p){(void)p;};
    return BASE_FUNC_NAME(acBoundaryCondition)(boundary, kernel, fields.data(), fields.size(), fields.data(), fields.size(), loader);
}

/** */
template <size_t n_ops>
static AcTaskGraph*
acGridBuildTaskGraph(const AcTaskDefinition (&ops)[n_ops])
{
    return BASE_FUNC_NAME(acGridBuildTaskGraph)(ops, n_ops);
}
static UNUSED AcTaskGraph*
acGridBuildTaskGraph(const std::vector<AcTaskDefinition> ops)
{
    return BASE_FUNC_NAME(acGridBuildTaskGraph)(ops.data(), ops.size());
}
#endif
#endif
#if AC_RUNTIME_COMPILATION
#include "astaroth_runtime_compilation.h"
#endif

#ifdef __cplusplus
#include <type_traits>


  template <typename P, typename V>
  void
  acPushToConfig(AcMeshInfo& config, P param, V val)
  {
	  static_assert(!std::is_same<P,int>::value);
	  static_assert(!std::is_same<P,AcReal>::value);
	  static_assert(!std::is_same<P,AcReal3>::value);
	  static_assert(!std::is_same<P,bool>::value);
	  static_assert(!std::is_same<P,AcBool3>::value);
	  if constexpr(IsCompParam(param))
	  {
	  	  config.run_consts.config[param] = val;
	  	  config.run_consts.is_loaded[param] = true;
	  }
	  else
		  config[param] = val;
  }

#endif
#include <string.h>

  static AcCompInfo UNUSED acInitCompInfo()
  {
	  AcCompInfo res;
	  memset(&res.is_loaded,0,sizeof(res.is_loaded));
	  memset(&res.config.bool_params,0,sizeof(res.config.bool_params));
	  memset(&res.config.bool3_params,0,sizeof(res.config.bool3_params));
	  return res;
  }
  static AcMeshInfo UNUSED acInitInfo()
  {
	  AcMeshInfo res;
    	  // memset reads the second parameter as a byte even though it says int in
          // the function declaration
    	  memset(&res, (uint8_t)0xFF, sizeof(res));
	  memset(&res.params.scalars.bool_params,0, sizeof(res.params.scalars.bool_params));
	  memset(&res.params.scalars.bool3_params,0,sizeof(res.params.scalars.bool3_params));
    	  //these are set to nullpointers for the users convenience that the user doesn't have to set them to null elsewhere
    	  //if they are present in the config then they are initialized correctly
	  memset(&res.params.arrays, 0, sizeof(res.params.arrays));

#if AC_MPI_ENABLED
	  res.comm = MPI_COMM_NULL;
#endif
	  res.run_consts = acInitCompInfo();
	  res.params.scalars.int3_params[AC_thread_block_loop_factors] = (int3){1,1,1};
	  res.params.scalars.int3_params[AC_max_tpb_for_reduce_kernels] = (int3){-1,8,8};
	  return res;
  }
  static AcMesh UNUSED acInitMesh()
  {
	  AcMesh res;
	  res.info = acInitInfo();
	  return res;
  }
