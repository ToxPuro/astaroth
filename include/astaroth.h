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
#define FUNC_DEFINE(return_type, func_name, ...) static return_type (*func_name) __VA_ARGS__
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

FUNC_DEFINE(int3, acGetLocalNN, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetLocalMM, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetGridNN, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetGridMM, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetMinNN, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(int3, acGetGridMaxNN, (const AcMeshInfo info));
FUNC_DEFINE(AcReal3, acGetLengths, (const AcMeshInfo info));


static inline size_t
acVertexBufferSize(const AcMeshInfo info)
{
    const int3 mm = acGetLocalMM(info);
    return as_size_t(mm.x)*as_size_t(mm.y)*as_size_t(mm.z);
}
static inline size_t
acGridVertexBufferSize(const AcMeshInfo info)
{
    const int3 mm = acGetGridMM(info);
    return as_size_t(mm.x)*as_size_t(mm.y)*as_size_t(mm.z);
}

static inline int3
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
    const int3 nn = acGetLocalNN(info);
    return as_size_t(nn.x)*as_size_t(nn.y)*as_size_t(nn.z);
}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info)
{
    return sizeof(AcReal) * acVertexBufferCompdomainSize(info);
}



typedef struct {
    int3 n0, n1;
    int3 m0, m1;
    int3 nn;
} AcMeshDims;




static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
   const int3 n0 = acGetMinNN(info);
   const int3 n1 = acGetMaxNN(info);
   const int3 m0 = (int3){0, 0, 0};
   const int3 m1 = acGetLocalMM(info);
   const int3 nn = acGetLocalNN(info);

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
   };
}

static inline AcMeshDims
acGetGridMeshDims(const AcMeshInfo info)
{
   const int3 n0 = acGetMinNN(info);
   const int3 n1 = acGetGridMaxNN(info);
   const int3 m0 = (int3){0, 0, 0};
   const int3 m1 = acGetGridMM(info);
   const int3 nn = acGetGridNN(info);

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
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
    auto mm = info[AC_mgrid];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    auto mm = info[AC_mlocal];
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#else
static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const int3 mm = acGetLocalMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
static inline size_t
acGridVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info)
{
    const int3 mm = acGetGridMM(info);
    return AC_INDEX_ORDER(i,j,k,mm.x,mm.y,mm.z);
}
#endif

static inline int3
acVertexBufferSpatialIdx(const size_t i, const AcMeshInfo info)
{
    const int3 mm = acGetLocalMM(info);

    return (int3){
        (int)i % mm.x,
        ((int)i % (mm.x * mm.y)) / mm.x,
        (int)i / (mm.x * mm.y),
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
} AcTaskDefinition;

/** TaskGraph is an opaque datatype containing information necessary to execute a set of
 * operations.*/
typedef struct AcTaskGraph AcTaskGraph;

#if __cplusplus
OVERLOADED_FUNC_DEFINE(AcTaskDefinition, acComputeWithParams,(const AcKernel kernel, Field fields_in[], const size_t num_fields_in,
                           Field fields_out[], const size_t num_fields_out,Profile profiles_in[], const size_t num_profiles_in, Profile profiles_out[], const size_t num_profiles_out, std::function<void(ParamLoadingInfo step_info)> loader));
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


/** */
FUNC_DEFINE(AcResult, acGridDestroyTaskGraph,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(AcResult, acGridExecuteTaskGraph,(AcTaskGraph* graph, const size_t n_iterations));
/** */
FUNC_DEFINE(AcResult, acGridFinalizeReduceLocal,(AcTaskGraph* graph));
/** */
FUNC_DEFINE(AcResult, acGridFinalizeReduce,(AcTaskGraph* graph));

/** */
FUNC_DEFINE(AcResult, acGridLaunchKernel,(const Stream stream, const AcKernel kernel, const int3 start,
                            const int3 end));


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
                                  const int3 start, const int3 end, const AcReal dt));
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
FUNC_DEFINE(AcResult,  acDeviceFinishReduce,(Device device, const Stream stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceInt,(Device device, const Stream stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output));
FUNC_DEFINE(AcResult,
acDevicePreprocessScratchPad,(Device device, const int variable, const AcType type,const AcReduceOp op));

/** */
FUNC_DEFINE(AcResult, acDeviceUpdate,(Device device, const AcMeshInfo info));

/** */
FUNC_DEFINE(AcDeviceKernelOutput, acDeviceGetKernelOutput,(const Device device));


/** */
FUNC_DEFINE(AcResult, acDeviceLaunchKernel,(const Device device, const Stream stream, const AcKernel kernel,
                              const int3 start, const int3 end));

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
FUNC_DEFINE(AcResult, acDeviceVolumeCopy,(const Device device, const Stream stream,const AcReal* in, const int3 in_offset, const int3 in_volume,AcReal* out, const int3 out_offset, const int3 out_volume));

/** */
FUNC_DEFINE(AcResult, acDeviceLoadPlateBuffer,(const Device device, int3 start, int3 end, const Stream stream,
                                 AcReal* buffer, int plate));

/** */
FUNC_DEFINE(AcResult, acDeviceStorePlateBuffer,(const Device device, int3 start, int3 end, const Stream stream, 
                                  AcReal* buffer, int plate));

/** */
FUNC_DEFINE(AcResult, acDeviceStoreIXYPlate,(const Device device, int3 start, int3 end, int src_offset, const Stream stream, 
                               AcMesh *host_mesh));

#if AC_RUNTIME_COMPILATION == 0
/** */
FUNC_DEFINE(AcResult, acDeviceGetVBApointers,(Device device, AcReal *vbapointer[2]));
#endif


/** */
AcResult acDeviceWriteMeshToDisk(const Device device, const VertexBufferHandle vtxbuf,
                                 const char* filepath);

/** */
AcMeshInfo acDeviceGetLocalConfig(const Device device);

/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
AcResult 
acHostUpdateBuiltinParams(AcMeshInfo* config);
AcResult 
acHostUpdateBuiltinCompParams(AcCompInfo* comp_config);



/** Creates a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));
/** Copies the VertexBuffers from src to dst*/
FUNC_DEFINE(AcResult, acHostMeshCopyVertexBuffers,(const AcMesh src, AcMesh dst));
/** Copies a host mesh to a new host mesh */
FUNC_DEFINE(AcResult, acHostMeshCopy,(const AcMesh src, AcMesh* dst));
/** Creates a mesh stored in host memory (size of the whole grid) */
FUNC_DEFINE(AcResult, acHostGridMeshCreate,(const AcMeshInfo mesh_info, AcMesh* mesh));

/** Checks that the loaded dynamic Astaroth is binary compatible with the loader */
FUNC_DEFINE(AcResult, acVerifyCompatibility, (const size_t mesh_size, const size_t mesh_info_size, const int num_reals, const int num_ints, const int num_bools, const int num_real_arrays, const int num_int_arrays, const int num_bool_arrays));

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

        LOAD_DSYM(acDeviceFinishReduceInt)
        LOAD_DSYM(acKernelFlushInt)
        LOAD_DSYM(acAnalysisGetKernelInfo)
        LOAD_DSYM(acDeviceSwapAllProfileBuffers)
#if AC_MPI_ENABLED
	LOAD_DSYM(BASE_FUNC_NAME(acBoundaryCondition))
	LOAD_DSYM(ac_MPI_Init)
	LOAD_DSYM(ac_MPI_Init_thread)
	LOAD_DSYM(ac_MPI_Finalize);
	*(void**)(&acGridMPIComm) = dlsym(handle,"acGridMPIComm");
	if(!acGridMPIComm) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridMPIComm");
	*(void**)(&acGridDecomposeMeshInfo) = dlsym(handle,"acGridDecomposeMeshInfo");
	if(!acGridDecomposeMeshInfo) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridDecomposeMeshInfo");
	*(void**)(&acGridGetLocalMeshInfo) = dlsym(handle,"acGridGetLocalMeshInfo");
	if(!acGridGetLocalMeshInfo) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridGetLocalMeshInfo");
	*(void**)(&acGridQuit) = dlsym(handle,"acGridQuit");
	if(!acGridQuit) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridQuit");
	*(void**)(&acGridGetDevice) = dlsym(handle,"acGridGetDevice");
	if(!acGridGetDevice) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridGetDevice");
	*(void**)(&acGridRandomize) = dlsym(handle,"acGridRandomize");
	if(!acGridRandomize) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridRandomize");
	*(void**)(&acGridSynchronizeStream) = dlsym(handle,"acGridSynchronizeStream");
	if(!acGridSynchronizeStream) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridSynchronizeStream");
	*(void**)(&acGridLoadScalarUniform) = dlsym(handle,"acGridLoadScalarUniform");
	if(!acGridLoadScalarUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadScalarUniform");
	*(void**)(&acGridLoadVectorUniform) = dlsym(handle,"acGridLoadVectorUniform");
	if(!acGridLoadVectorUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadVectorUniform");
	*(void**)(&acGridLoadIntUniform) = dlsym(handle,"acGridLoadIntUniform");
	if(!acGridLoadIntUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadIntUniform");
	*(void**)(&acGridLoadInt3Uniform) = dlsym(handle,"acGridLoadInt3Uniform");
	if(!acGridLoadInt3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadInt3Uniform");
	*(void**)(&acGridLoadMesh) = dlsym(handle,"acGridLoadMesh");
	if(!acGridLoadMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadMesh");
	*(void**)(&acGridStoreMesh) = dlsym(handle,"acGridStoreMesh");
	if(!acGridStoreMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridStoreMesh");
	*(void**)(&acGridIntegrate) = dlsym(handle,"acGridIntegrate");
	if(!acGridIntegrate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridIntegrate");
	*(void**)(&acGridSwapBuffers) = dlsym(handle,"acGridSwapBuffers");
	if(!acGridSwapBuffers) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridSwapBuffers");
	*(void**)(&acGridPeriodicBoundconds) = dlsym(handle,"acGridPeriodicBoundconds");
	if(!acGridPeriodicBoundconds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridPeriodicBoundconds");
	*(void**)(&acGridReduceScal) = dlsym(handle,"acGridReduceScal");
	if(!acGridReduceScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridReduceScal");
	*(void**)(&acGridReduceVec) = dlsym(handle,"acGridReduceVec");
	if(!acGridReduceVec) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridReduceVec");
	*(void**)(&acGridReduceVecScal) = dlsym(handle,"acGridReduceVecScal");
	if(!acGridReduceVecScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridReduceVecScal");
	*(void**)(&acGridAccessMeshOnDiskSynchronous) = dlsym(handle,"acGridAccessMeshOnDiskSynchronous");
	if(!acGridAccessMeshOnDiskSynchronous) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridAccessMeshOnDiskSynchronous");
	*(void**)(&acGridDiskAccessLaunch) = dlsym(handle,"acGridDiskAccessLaunch");
	if(!acGridDiskAccessLaunch) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridDiskAccessLaunch");
	*(void**)(&acGridWriteSlicesToDiskLaunch) = dlsym(handle,"acGridWriteSlicesToDiskLaunch");
	if(!acGridWriteSlicesToDiskLaunch) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridWriteSlicesToDiskLaunch");
	*(void**)(&acGridWriteSlicesToDiskCollectiveSynchronous) = dlsym(handle,"acGridWriteSlicesToDiskCollectiveSynchronous");
	if(!acGridWriteSlicesToDiskCollectiveSynchronous) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridWriteSlicesToDiskCollectiveSynchronous");
	*(void**)(&acGridWriteMeshToDiskLaunch) = dlsym(handle,"acGridWriteMeshToDiskLaunch");
	if(!acGridWriteMeshToDiskLaunch) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridWriteMeshToDiskLaunch");
	*(void**)(&acGridDiskAccessSync) = dlsym(handle,"acGridDiskAccessSync");
	if(!acGridDiskAccessSync) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridDiskAccessSync");
	*(void**)(&acGridReadVarfileToMesh) = dlsym(handle,"acGridReadVarfileToMesh");
	if(!acGridReadVarfileToMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridReadVarfileToMesh");
	*(void**)(&acGridAccessMeshOnDiskSynchronousDistributed) = dlsym(handle,"acGridAccessMeshOnDiskSynchronousDistributed");
	if(!acGridAccessMeshOnDiskSynchronousDistributed) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridAccessMeshOnDiskSynchronousDistributed");
	*(void**)(&acGridAccessMeshOnDiskSynchronousCollective) = dlsym(handle,"acGridAccessMeshOnDiskSynchronousCollective");
	if(!acGridAccessMeshOnDiskSynchronousCollective) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridAccessMeshOnDiskSynchronousCollective");
	*(void**)(&BASE_FUNC_NAME(acComputeWithParams)) = dlsym(handle,"acComputeWithParams");
	*(void**)(&BASE_FUNC_NAME(acCompute)) = dlsym(handle,"acCompute");
	*(void**)(&BASE_FUNC_NAME(acHaloExchange)) = dlsym(handle,"acHaloExchange");
	*(void**)(&acGridGetDefaultTaskGraph) = dlsym(handle,"acGridGetDefaultTaskGraph");
	if(!acGridGetDefaultTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridGetDefaultTaskGraph");
	*(void**)(&acGridTaskGraphHasPeriodicBoundcondsX) = dlsym(handle,"acGridTaskGraphHasPeriodicBoundcondsX");
	if(!acGridTaskGraphHasPeriodicBoundcondsX) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridTaskGraphHasPeriodicBoundcondsX");
	*(void**)(&acGridTaskGraphHasPeriodicBoundcondsY) = dlsym(handle,"acGridTaskGraphHasPeriodicBoundcondsY");
	if(!acGridTaskGraphHasPeriodicBoundcondsY) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridTaskGraphHasPeriodicBoundcondsY");
	*(void**)(&acGridTaskGraphHasPeriodicBoundcondsZ) = dlsym(handle,"acGridTaskGraphHasPeriodicBoundcondsZ");
	if(!acGridTaskGraphHasPeriodicBoundcondsZ) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridTaskGraphHasPeriodicBoundcondsZ");
	*(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle,"acGridBuildTaskGraph");
	*(void**)(&acGridDestroyTaskGraph) = dlsym(handle,"acGridDestroyTaskGraph");
	*(void**)(&(acGetDSLTaskGraph)) = dlsym(handle,"acGetDSLTaskGraph");
	if(!acGetDSLTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetDSLTaskGraph");
	if(!acGridDestroyTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridDestroyTaskGraph");
	*(void**)(&acGridExecuteTaskGraph) = dlsym(handle,"acGridExecuteTaskGraph");
	if(!acGridExecuteTaskGraph) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridExecuteTaskGraph");
	*(void**)(&acGridFinalizeReduceLocal) = dlsym(handle,"acGridFinalizeReduceLocal");
	if(!acGridFinalizeReduceLocal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridFinalizeReduceLocal");
	*(void**)(&acGridFinalizeReduce) = dlsym(handle,"acGridFinalizeReduce");
	if(!acGridFinalizeReduce) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridFinalizeReduce");
	*(void**)(&acGridLaunchKernel) = dlsym(handle,"acGridLaunchKernel");
	if(!acGridLaunchKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLaunchKernel");
	*(void**)(&acGridLoadStencil) = dlsym(handle,"acGridLoadStencil");
	if(!acGridLoadStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadStencil");
	*(void**)(&acGridStoreStencil) = dlsym(handle,"acGridStoreStencil");
	if(!acGridStoreStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridStoreStencil");
	*(void**)(&acGridLoadStencils) = dlsym(handle,"acGridLoadStencils");
	if(!acGridLoadStencils) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridLoadStencils");
	*(void**)(&acGridStoreStencils) = dlsym(handle,"acGridStoreStencils");
	if(!acGridStoreStencils) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGridStoreStencils");
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
	LOAD_DSYM(acHostMeshCopy)
#include "device_load_uniform_loads.h"
	*(void**)(&acGetKernelId) = dlsym(handle,"acGetKernelId");
	if(!acGetKernelId) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetKernelId");
	*(void**)(&acGetKernelIdByName) = dlsym(handle,"acGetKernelIdByName");
	if(!acGetKernelIdByName) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetKernelIdByName");
	*(void**)(&acCheckDeviceAvailability) = dlsym(handle,"acCheckDeviceAvailability");
	if(!acCheckDeviceAvailability) fprintf(stderr,"Astaroth error: was not able to load %s\n","acCheckDeviceAvailability");
	*(void**)(&acGetNumDevicesPerNode) = dlsym(handle,"acGetNumDevicesPerNode");
	if(!acGetNumDevicesPerNode) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetNumDevicesPerNode");
	*(void**)(&acGetNumFields) = dlsym(handle,"acGetNumFields");
	if(!acGetNumFields) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetNumFields");
	*(void**)(&acGetFieldHandle) = dlsym(handle,"acGetFieldHandle");
	if(!acGetFieldHandle) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetFieldHandle");
	*(void**)(&acGetNode) = dlsym(handle,"acGetNode");
	if(!acGetNode) fprintf(stderr,"Astaroth error: was not able to load %s\n","acGetNode");


	*(void**)(&acNodeCreate) = dlsym(handle,"acNodeCreate");
	if(!acNodeCreate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeCreate");
	*(void**)(&acNodeDestroy) = dlsym(handle,"acNodeDestroy");
	if(!acNodeDestroy) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeDestroy");
	*(void**)(&acNodePrintInfo) = dlsym(handle,"acNodePrintInfo");
	if(!acNodePrintInfo) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodePrintInfo");
	*(void**)(&acNodeQueryDeviceConfiguration) = dlsym(handle,"acNodeQueryDeviceConfiguration");
	if(!acNodeQueryDeviceConfiguration) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeQueryDeviceConfiguration");
	*(void**)(&acNodeAutoOptimize) = dlsym(handle,"acNodeAutoOptimize");
	if(!acNodeAutoOptimize) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeAutoOptimize");
	*(void**)(&acNodeSynchronizeStream) = dlsym(handle,"acNodeSynchronizeStream");
	if(!acNodeSynchronizeStream) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeSynchronizeStream");
	*(void**)(&acNodeSynchronizeVertexBuffer) = dlsym(handle,"acNodeSynchronizeVertexBuffer");
	if(!acNodeSynchronizeVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeSynchronizeVertexBuffer");
	*(void**)(&acNodeSynchronizeMesh) = dlsym(handle,"acNodeSynchronizeMesh");
	if(!acNodeSynchronizeMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeSynchronizeMesh");
	*(void**)(&acNodeSwapBuffers) = dlsym(handle,"acNodeSwapBuffers");
	if(!acNodeSwapBuffers) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeSwapBuffers");
	*(void**)(&acNodeLoadConstant) = dlsym(handle,"acNodeLoadConstant");
	if(!acNodeLoadConstant) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadConstant");
	*(void**)(&acNodeLoadVertexBufferWithOffset) = dlsym(handle,"acNodeLoadVertexBufferWithOffset");
	if(!acNodeLoadVertexBufferWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadVertexBufferWithOffset");
	*(void**)(&acNodeLoadMeshWithOffset) = dlsym(handle,"acNodeLoadMeshWithOffset");
	if(!acNodeLoadMeshWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadMeshWithOffset");
	*(void**)(&acNodeLoadVertexBuffer) = dlsym(handle,"acNodeLoadVertexBuffer");
	if(!acNodeLoadVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadVertexBuffer");
	*(void**)(&acNodeLoadMesh) = dlsym(handle,"acNodeLoadMesh");
	if(!acNodeLoadMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadMesh");
	*(void**)(&acNodeSetVertexBuffer) = dlsym(handle,"acNodeSetVertexBuffer");
	if(!acNodeSetVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeSetVertexBuffer");
	*(void**)(&acNodeStoreVertexBufferWithOffset) = dlsym(handle,"acNodeStoreVertexBufferWithOffset");
	if(!acNodeStoreVertexBufferWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStoreVertexBufferWithOffset");
	*(void**)(&acNodeStoreMeshWithOffset) = dlsym(handle,"acNodeStoreMeshWithOffset");
	if(!acNodeStoreMeshWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStoreMeshWithOffset");
	*(void**)(&acNodeStoreVertexBuffer) = dlsym(handle,"acNodeStoreVertexBuffer");
	if(!acNodeStoreVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStoreVertexBuffer");
	*(void**)(&acNodeStoreMesh) = dlsym(handle,"acNodeStoreMesh");
	if(!acNodeStoreMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStoreMesh");
	*(void**)(&acNodeIntegrateSubstep) = dlsym(handle,"acNodeIntegrateSubstep");
	if(!acNodeIntegrateSubstep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeIntegrateSubstep");
	*(void**)(&acNodeIntegrate) = dlsym(handle,"acNodeIntegrate");
	if(!acNodeIntegrate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeIntegrate");
	*(void**)(&acNodeIntegrateGBC) = dlsym(handle,"acNodeIntegrateGBC");
	if(!acNodeIntegrateGBC) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeIntegrateGBC");
	*(void**)(&acNodePeriodicBoundcondStep) = dlsym(handle,"acNodePeriodicBoundcondStep");
	if(!acNodePeriodicBoundcondStep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodePeriodicBoundcondStep");
	*(void**)(&acNodePeriodicBoundconds) = dlsym(handle,"acNodePeriodicBoundconds");
	if(!acNodePeriodicBoundconds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodePeriodicBoundconds");
	*(void**)(&acNodeGeneralBoundcondStep) = dlsym(handle,"acNodeGeneralBoundcondStep");
	if(!acNodeGeneralBoundcondStep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeGeneralBoundcondStep");
	*(void**)(&acNodeGeneralBoundconds) = dlsym(handle,"acNodeGeneralBoundconds");
	if(!acNodeGeneralBoundconds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeGeneralBoundconds");
	*(void**)(&acNodeReduceScal) = dlsym(handle,"acNodeReduceScal");
	if(!acNodeReduceScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeReduceScal");
	*(void**)(&acNodeReduceVec) = dlsym(handle,"acNodeReduceVec");
	if(!acNodeReduceVec) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeReduceVec");
	*(void**)(&acNodeReduceVecScal) = dlsym(handle,"acNodeReduceVecScal");
	if(!acNodeReduceVecScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeReduceVecScal");
	*(void**)(&acNodeLoadPlate) = dlsym(handle,"acNodeLoadPlate");
	if(!acNodeLoadPlate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadPlate");
	*(void**)(&acNodeStorePlate) = dlsym(handle,"acNodeStorePlate");
	if(!acNodeStorePlate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStorePlate");
	*(void**)(&acNodeStoreIXYPlate) = dlsym(handle,"acNodeStoreIXYPlate");
	if(!acNodeStoreIXYPlate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeStoreIXYPlate");
	*(void**)(&acNodeLoadPlateXcomp) = dlsym(handle,"acNodeLoadPlateXcomp");
	if(!acNodeLoadPlateXcomp) fprintf(stderr,"Astaroth error: was not able to load %s\n","acNodeLoadPlateXcomp");
	*(void**)(&acDeviceCreate) = dlsym(handle,"acDeviceCreate");
	if(!acDeviceCreate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceCreate");
	*(void**)(&acDeviceDestroy) = dlsym(handle,"acDeviceDestroy");
	if(!acDeviceDestroy) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceDestroy");
	*(void**)(&acDeviceResetMesh) = dlsym(handle,"acDeviceResetMesh");
	if(!acDeviceResetMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceResetMesh");
	*(void**)(&acDevicePrintInfo) = dlsym(handle,"acDevicePrintInfo");
	if(!acDevicePrintInfo) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDevicePrintInfo");
	*(void**)(&acDeviceSynchronizeStream) = dlsym(handle,"acDeviceSynchronizeStream");
	if(!acDeviceSynchronizeStream) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceSynchronizeStream");
	*(void**)(&acDeviceSwapBuffer) = dlsym(handle,"acDeviceSwapBuffer");
	if(!acDeviceSwapBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceSwapBuffer");
	*(void**)(&acDeviceSwapBuffers) = dlsym(handle,"acDeviceSwapBuffers");
	if(!acDeviceSwapBuffers) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceSwapBuffers");
	*(void**)(&acDeviceLoadScalarUniform) = dlsym(handle,"acDeviceLoadScalarUniform");
	if(!acDeviceLoadScalarUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadScalarUniform");
	*(void**)(&acDeviceLoadVectorUniform) = dlsym(handle,"acDeviceLoadVectorUniform");
	if(!acDeviceLoadVectorUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadVectorUniform");
	*(void**)(&acDeviceLoadIntUniform) = dlsym(handle,"acDeviceLoadIntUniform");
	if(!acDeviceLoadIntUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadIntUniform");
	*(void**)(&acDeviceLoadBoolUniform) = dlsym(handle,"acDeviceLoadBoolUniform");
	if(!acDeviceLoadBoolUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadBoolUniform");
	*(void**)(&acDeviceLoadInt3Uniform) = dlsym(handle,"acDeviceLoadInt3Uniform");
	if(!acDeviceLoadInt3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadInt3Uniform");
	*(void**)(&acDeviceStoreScalarUniform) = dlsym(handle,"acDeviceStoreScalarUniform");
	if(!acDeviceStoreScalarUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreScalarUniform");
	*(void**)(&acDeviceStoreVectorUniform) = dlsym(handle,"acDeviceStoreVectorUniform");
	if(!acDeviceStoreVectorUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreVectorUniform");
	*(void**)(&acDeviceStoreIntUniform) = dlsym(handle,"acDeviceStoreIntUniform");
	if(!acDeviceStoreIntUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreIntUniform");
	*(void**)(&acDeviceStoreBoolUniform) = dlsym(handle,"acDeviceStoreBoolUniform");
	if(!acDeviceStoreBoolUniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreBoolUniform");
	*(void**)(&acDeviceStoreInt3Uniform) = dlsym(handle,"acDeviceStoreInt3Uniform");
	if(!acDeviceStoreInt3Uniform) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreInt3Uniform");
	*(void**)(&acDeviceLoadMeshInfo) = dlsym(handle,"acDeviceLoadMeshInfo");
	if(!acDeviceLoadMeshInfo) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadMeshInfo");
	*(void**)(&acDeviceLoadVertexBufferWithOffset) = dlsym(handle,"acDeviceLoadVertexBufferWithOffset");
	if(!acDeviceLoadVertexBufferWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadVertexBufferWithOffset");
	*(void**)(&acDeviceLoadMeshWithOffset) = dlsym(handle,"acDeviceLoadMeshWithOffset");
	if(!acDeviceLoadMeshWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadMeshWithOffset");
	*(void**)(&acDeviceLoadVertexBuffer) = dlsym(handle,"acDeviceLoadVertexBuffer");
	if(!acDeviceLoadVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadVertexBuffer");
	*(void**)(&acDeviceLoadMesh) = dlsym(handle,"acDeviceLoadMesh");
	if(!acDeviceLoadMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadMesh");
	*(void**)(&acDeviceSetVertexBuffer) = dlsym(handle,"acDeviceSetVertexBuffer");
	if(!acDeviceSetVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceSetVertexBuffer");
	*(void**)(&acDeviceFlushOutputBuffers) = dlsym(handle,"acDeviceFlushOutputBuffers");
	if(!acDeviceFlushOutputBuffers) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceFlushOutputBuffers");
	*(void**)(&acDeviceStoreVertexBufferWithOffset) = dlsym(handle,"acDeviceStoreVertexBufferWithOffset");
	if(!acDeviceStoreVertexBufferWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreVertexBufferWithOffset");
	*(void**)(&acDeviceGetConfig) = dlsym(handle,"acDeviceGetConfig");
	if(!acDeviceGetConfig) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceGetConfig");
	*(void**)(&acDeviceGetKernelInputParamsObject) = dlsym(handle,"acDeviceGetKernelInputParamsObject");
	if(!acDeviceGetKernelInputParamsObject) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceGetKernelInputParamsObject");
	*(void**)(&acDeviceStoreMeshWithOffset) = dlsym(handle,"acDeviceStoreMeshWithOffset");
	if(!acDeviceStoreMeshWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreMeshWithOffset");
	*(void**)(&acDeviceStoreVertexBuffer) = dlsym(handle,"acDeviceStoreVertexBuffer");
	if(!acDeviceStoreVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreVertexBuffer");
	*(void**)(&acDeviceStoreMesh) = dlsym(handle,"acDeviceStoreMesh");
	if(!acDeviceStoreMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreMesh");
	*(void**)(&acDeviceTransferVertexBufferWithOffset) = dlsym(handle,"acDeviceTransferVertexBufferWithOffset");
	if(!acDeviceTransferVertexBufferWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceTransferVertexBufferWithOffset");
	*(void**)(&acDeviceTransferMeshWithOffset) = dlsym(handle,"acDeviceTransferMeshWithOffset");
	if(!acDeviceTransferMeshWithOffset) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceTransferMeshWithOffset");
	*(void**)(&acDeviceTransferVertexBuffer) = dlsym(handle,"acDeviceTransferVertexBuffer");
	if(!acDeviceTransferVertexBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceTransferVertexBuffer");
	*(void**)(&acDeviceTransferMesh) = dlsym(handle,"acDeviceTransferMesh");
	if(!acDeviceTransferMesh) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceTransferMesh");
	*(void**)(&acDeviceIntegrateSubstep) = dlsym(handle,"acDeviceIntegrateSubstep");
	if(!acDeviceIntegrateSubstep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceIntegrateSubstep");
	*(void**)(&acDevicePeriodicBoundcondStep) = dlsym(handle,"acDevicePeriodicBoundcondStep");
	if(!acDevicePeriodicBoundcondStep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDevicePeriodicBoundcondStep");
	*(void**)(&acDevicePeriodicBoundconds) = dlsym(handle,"acDevicePeriodicBoundconds");
	if(!acDevicePeriodicBoundconds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDevicePeriodicBoundconds");
	*(void**)(&acDeviceGeneralBoundcondStep) = dlsym(handle,"acDeviceGeneralBoundcondStep");
	if(!acDeviceGeneralBoundcondStep) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceGeneralBoundcondStep");
	*(void**)(&acDeviceGeneralBoundconds) = dlsym(handle,"acDeviceGeneralBoundconds");
	if(!acDeviceGeneralBoundconds) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceGeneralBoundconds");
	LOAD_DSYM(acDeviceReduceScalNoPostProcessing)
	*(void**)(&acDeviceReduceScal) = dlsym(handle,"acDeviceReduceScal");
	if(!acDeviceReduceScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceReduceScal");
	LOAD_DSYM(acDeviceReduceVecNoPostProcessing)
	*(void**)(&acDeviceReduceVec) = dlsym(handle,"acDeviceReduceVec");
	if(!acDeviceReduceVec) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceReduceVec");
	LOAD_DSYM(acDeviceReduceVecScalNoPostProcessing)
	*(void**)(&acDeviceReduceVecScal) = dlsym(handle,"acDeviceReduceVecScal");
	if(!acDeviceReduceVecScal) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceReduceVecScal");
	*(void**)(&acDeviceFinishReduce) = dlsym(handle,"acDeviceFinishReduce");
	if(!acDeviceFinishReduce) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceFinishReduce");
	*(void**)(&acDeviceUpdate) = dlsym(handle,"acDeviceUpdate");
	if(!acDeviceUpdate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceUpdate");
	*(void**)(&acDeviceGetKernelOutput) = dlsym(handle,"acDeviceGetKernelOutput");
	if(!acDeviceGetKernelOutput) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceGetKernelOutput");
	*(void**)(&acDeviceLaunchKernel) = dlsym(handle,"acDeviceLaunchKernel");
	if(!acDeviceLaunchKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLaunchKernel");
	*(void**)(&acDeviceBenchmarkKernel) = dlsym(handle,"acDeviceBenchmarkKernel");
	if(!acDeviceBenchmarkKernel) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceBenchmarkKernel");
	*(void**)(&acDeviceLoadStencil) = dlsym(handle,"acDeviceLoadStencil");
	if(!acDeviceLoadStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadStencil");
	*(void**)(&acDeviceLoadStencils) = dlsym(handle,"acDeviceLoadStencils");
	if(!acDeviceLoadStencils) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadStencils");
	*(void**)(&acDeviceLoadStencilsFromConfig) = dlsym(handle,"acDeviceLoadStencilsFromConfig");
	if(!acDeviceLoadStencilsFromConfig) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadStencilsFromConfig");
	*(void**)(&acDeviceStoreStencil) = dlsym(handle,"acDeviceStoreStencil");
	if(!acDeviceStoreStencil) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreStencil");
	*(void**)(&acDeviceVolumeCopy) = dlsym(handle,"acDeviceVolumeCopy");
	if(!acDeviceVolumeCopy) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceVolumeCopy");
	*(void**)(&acDeviceLoadPlateBuffer) = dlsym(handle,"acDeviceLoadPlateBuffer");
	if(!acDeviceLoadPlateBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceLoadPlateBuffer");
	*(void**)(&acDeviceStorePlateBuffer) = dlsym(handle,"acDeviceStorePlateBuffer");
	if(!acDeviceStorePlateBuffer) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStorePlateBuffer");
	*(void**)(&acDeviceStoreIXYPlate) = dlsym(handle,"acDeviceStoreIXYPlate");
	if(!acDeviceStoreIXYPlate) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDeviceStoreIXYPlate");
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
	*(void**)(&acHostMeshDestroy) = dlsym(handle,"acHostMeshDestroy");
	if(!acHostMeshDestroy) fprintf(stderr,"Astaroth error: was not able to load %s\n","acHostMeshDestroy");
	*(void**)(&acLogFromRootProc) = dlsym(handle,"acLogFromRootProc");
	if(!acLogFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acLogFromRootProc");
	*(void**)(&acVA_LogFromRootProc) = dlsym(handle,"acVA_LogFromRootProc");
	if(!acVA_LogFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVA_LogFromRootProc");
	*(void**)(&acVerboseLogFromRootProc) = dlsym(handle,"acVerboseLogFromRootProc");
	if(!acVerboseLogFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVerboseLogFromRootProc");
	*(void**)(&acVA_VerboseLogFromRootProc) = dlsym(handle,"acVA_VerboseLogFromRootProc");
	if(!acVA_VerboseLogFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVA_VerboseLogFromRootProc");
	*(void**)(&acDebugFromRootProc) = dlsym(handle,"acDebugFromRootProc");
	if(!acDebugFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acDebugFromRootProc");
	*(void**)(&acVA_DebugFromRootProc) = dlsym(handle,"acVA_DebugFromRootProc");
	if(!acVA_DebugFromRootProc) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVA_DebugFromRootProc");
	*(void**)(&acVerifyCompatibility) = dlsym(handle,"acVerifyCompatibility");
	if(!acVerifyCompatibility) fprintf(stderr,"Astaroth error: was not able to load %s\n","acVerifyCompatibility");
//#ifdef __cplusplus
//	return AcLibHandle(handle);
//#else
//	return handle;
//#endif
	const AcResult is_compatible = acVerifyCompatibility(sizeof(AcMesh), sizeof(AcMeshInfo), NUM_REAL_PARAMS, NUM_INT_PARAMS, NUM_BOOL_PARAMS, NUM_REAL_ARRAYS, NUM_INT_ARRAYS, NUM_BOOL_ARRAYS);
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

AcMeshOrder
acGetMeshOrderForProfile(const AcProfileType type);
AcShape
acGetTransposeBufferShape(const AcMeshOrder order, const AcMeshDims dims);
AcShape
acGetReductionShape(const AcProfileType type, const AcMeshDims dims);
AcResult
acReduceProfile(const Profile prof, const AcMeshDims dims, const AcReal* src, AcReal** tmp, size_t* tmp_size, AcReal* dst, const cudaStream_t stream);

void acBufferDestroy(AcBuffer* buffer);

AcResult acBufferMigrate(const AcBuffer in, AcBuffer* out);
AcBuffer acBufferCopy(const AcBuffer in, const bool on_device);

#ifdef __cplusplus
} // extern "C"
#endif


#ifdef __cplusplus


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
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0 ,loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::vector<Profile> profile_in, std::vector<Profile> profile_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), profile_in.data(), profile_in.size(), profile_out.data(), profile_out.size(), loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out)
{
    return BASE_FUNC_NAME(acCompute)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields.data(), fields.size(), fields.data(), fields.size(), NULL, 0 , NULL, 0, loader);
}


template <size_t num_fields_in, size_t num_fields_out>
static AcTaskDefinition
acCompute(AcKernel kernel, Field (&fields_in)[num_fields_in], Field (&fields_out)[num_fields_out], std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in, num_fields_in, fields_out, num_fields_out, NULL, 0, NULL, 0, loader);
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

  static AcCompInfo __attribute__((unused)) acInitCompInfo()
  {
	  AcCompInfo res;
	  memset(&res.is_loaded,0,sizeof(res.is_loaded));
	  return res;
  }
  static AcMeshInfo __attribute__((unused)) acInitInfo()
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
	  return res;
  }
