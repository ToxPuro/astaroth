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

#include "astaroth_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int3 acConstructInt3Param(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                          const AcMeshInfo info);

static inline AcReal3
acConstructReal3Param(const AcRealParam a, const AcRealParam b, const AcRealParam c,
                     const AcMeshInfo info)
{
    return (AcReal3){
        info.real_params[a],
        info.real_params[b],
        info.real_params[c],
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
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
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
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
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


FUNC_DEFINE(AcResult, acAnalysisGetKernelInfo,(const AcMeshInfo info, KernelAnalysisInfo* dst));
	



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



/** Prints a list of initial condition condition types */
void acQueryInitcondtypes(void);

/** Prints a list of int parameters */
void acQueryIntparams(void);

/** Prints a list of int3 parameters */
void acQueryInt3params(void);

/** Prints a list of real parameters */
void acQueryRealparams(void);

/** Prints a list of real3 parameters */
void acQueryReal3params(void);

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
void acQueryKernels(void);

static inline void
acPrintIntParam(const AcIntParam a, const AcMeshInfo info)
{
    printf("%s: %d\n", intparam_names[a], info.int_params[a]);
}

void acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                      const AcMeshInfo info);

static inline void
acPrintInt3Param(const AcInt3Param a, const AcMeshInfo info)
{
    const int3 vec = info.int3_params[a];
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
#include "astaroth_grid.h"
/*
 * =============================================================================
 * Node interface
 * =============================================================================
 */

#include "astaroth_node.h"

/*
 * =============================================================================
 * Device interface
 * =============================================================================
 */

#include "astaroth_device.h"

/*
 * =============================================================================
 * Legacy interface
 * =============================================================================
 */
#include "astaroth_legacy.h"
/*
 * =============================================================================
 * Helper functions
 * =============================================================================
 */
AcResult 
acHostUpdateParams(AcMeshInfo* config);

AcResult 
acHostUpdateCompParams(AcMeshInfo* config);



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
FUNC_DEFINE(AcResult, acVerifyCompatibility, (const size_t mesh_size, const size_t mesh_info_size, const size_t comp_info_size, const int num_reals, const int num_ints, const int num_bools, const int num_real_arrays, const int num_int_arrays, const int num_bool_arrays));

/** Randomizes a host mesh */
FUNC_DEFINE(AcResult, acHostMeshRandomize,(AcMesh* mesh));
/** Randomizes a host mesh (uses n[xyz]grid params)*/
FUNC_DEFINE(AcResult, acHostGridMeshRandomize,(AcMesh* mesh));

/** Destroys a mesh stored in host memory */
FUNC_DEFINE(AcResult, acHostMeshDestroy,(AcMesh* mesh));

FUNC_DEFINE(void, acStoreConfig,(const AcMeshInfo info, const char* filename));

/** Prints all parameters inside AcMeshInfo */
static inline void
acPrintMeshInfo(const AcMeshInfo config)
{
    acStoreConfig(config,NULL);
}

/** Sets the dimensions of the computational grid to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */
AcResult acSetGridMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

/** Sets the dimensions of the computational subdomain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */

AcResult acSetLocalMeshDims(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info);

/*
 * =============================================================================
 * Logging functions
 * =============================================================================
 */

/* Log a message with a timestamp from the root proc (if pid == 0) */
void acLogFromRootProc(const int pid, const char* msg, ...);
void acVA_LogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) if the build flag VERBOSE is on
 */
void acVerboseLogFromRootProc(const int pid, const char* msg, ...);
void acVA_VerboseLogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) in a debug build */
void acDebugFromRootProc(const int pid, const char* msg, ...);
void acVA_DebugFromRootProc(const int pid, const char* msg, va_list arg);


#include "device_set_input_decls.h"
#include "device_get_output_decls.h"
#include "device_get_input_decls.h"
#include "get_vtxbufs_declares.h"

#if AC_RUNTIME_COMPILATION
#include "astaroth_lib.h"

#define LOAD_DSYM(FUNC_NAME,STREAM) *(void**)(&FUNC_NAME) = dlsym(handle,#FUNC_NAME); \
			     if(!FUNC_NAME && STREAM) fprintf(STREAM,"Astaroth error: was not able to load %s\n",#FUNC_NAME);

  static AcResult __attribute__((unused)) acLoadLibrary(FILE* stream)
  {
	kernelsLibHandle=acLoadRunTime(stream);
 	void* handle = dlopen(runtime_astaroth_path,RTLD_NOW);
	if (!handle)
	{
    		fprintf(stderr,"%s","Fatal error was not able to load Astaroth\n"); 
		fprintf(stderr,"Error message: %s\n",dlerror());
		exit(EXIT_FAILURE);
	}
	astarothLibHandle=handle;

	LOAD_DSYM(acDeviceGetVertexBufferPtrs,stream)
	LOAD_DSYM(acDeviceGetLocalConfig,stream)
        LOAD_DSYM(acDeviceFinishReduceInt,stream) 
	LOAD_DSYM(acKernelFlushInt,stream) 
	LOAD_DSYM(acAnalysisGetKernelInfo,stream)
        LOAD_DSYM(acDeviceSwapAllProfileBuffers,stream)
#if AC_MPI_ENABLED
	LOAD_DSYM(BASE_FUNC_NAME(acBoundaryCondition),stream)
	LOAD_DSYM(ac_MPI_Init,stream)
	LOAD_DSYM(ac_MPI_Init_thread,stream)
	LOAD_DSYM(ac_MPI_Finalize,stream);
	LOAD_DSYM(acGridInitialized,stream);
	LOAD_DSYM(acGridMPIComm,stream);
	LOAD_DSYM(acGridMPISubComms,stream);
	LOAD_DSYM(acGridDecomposeMeshInfo,stream);
	LOAD_DSYM(acGridGetLocalMeshInfo,stream);
	LOAD_DSYM(acGridQuit,stream);
	LOAD_DSYM(acGridGetDevice,stream);
	LOAD_DSYM(acGridRandomize,stream);
	LOAD_DSYM(acGridSynchronizeStream,stream);
	LOAD_DSYM(acGridLoadScalarUniform,stream);
	LOAD_DSYM(acGridLoadVectorUniform,stream);
	LOAD_DSYM(acGridLoadIntUniform,stream);
	LOAD_DSYM(acGridLoadInt3Uniform,stream);
	LOAD_DSYM(acGridLoadMesh,stream);
	LOAD_DSYM(acGridStoreMesh,stream);
	LOAD_DSYM(acGridIntegrate,stream);
	LOAD_DSYM(acGridSwapBuffers,stream);
	LOAD_DSYM(acGridPeriodicBoundconds,stream);
	LOAD_DSYM(acGridReduceScal,stream);
	LOAD_DSYM(acGridReduceVec,stream);
	LOAD_DSYM(acGridReduceVecScal,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronous,stream);
	LOAD_DSYM(acGridDiskAccessLaunch,stream);
	LOAD_DSYM(acGridWriteSlicesToDiskLaunch,stream);
	LOAD_DSYM(acGridWriteSlicesToDiskCollectiveSynchronous,stream);
	LOAD_DSYM(acGridWriteMeshToDiskLaunch,stream);
	LOAD_DSYM(acGridDiskAccessSync,stream);
	LOAD_DSYM(acGridReadVarfileToMesh,stream);
	*(void**)(&BASE_FUNC_NAME(acComputeWithParams)) = dlsym(handle,"acComputeWithParams");
	*(void**)(&BASE_FUNC_NAME(acCompute)) = dlsym(handle,"acCompute");
	*(void**)(&BASE_FUNC_NAME(acHaloExchange)) = dlsym(handle,"acHaloExchange");
	*(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle,"acGridBuildTaskGraph");
	LOAD_DSYM(acGridDestroyTaskGraph,stream);
	LOAD_DSYM(acGridClearTaskGraphCache,stream);
	LOAD_DSYM(acGetDSLTaskGraph,stream);
	LOAD_DSYM(acGetOptimizedDSLTaskGraph,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousDistributed,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousCollective,stream);
	LOAD_DSYM(acGridGetDefaultTaskGraph,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsX,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsY,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsZ,stream);
	LOAD_DSYM(acGridExecuteTaskGraph,stream);
	LOAD_DSYM(acGridExecuteTaskGraphBase,stream);
	LOAD_DSYM(acGridFinalizeReduceLocal,stream);
	LOAD_DSYM(acGridFinalizeReduce,stream);
	LOAD_DSYM(acGridLaunchKernel,stream);
	LOAD_DSYM(acGridLoadStencil,stream);
	LOAD_DSYM(acGridStoreStencil,stream);
	LOAD_DSYM(acGridLoadStencils,stream);
	LOAD_DSYM(acGridStoreStencils,stream);
	LOAD_DSYM(acGridInitBase,stream);
#endif
	LOAD_DSYM(acGetLocalNN,stream)
	LOAD_DSYM(acGetLocalMM,stream)
	LOAD_DSYM(acGetGridNN,stream)
	LOAD_DSYM(acGetGridMM,stream)
	LOAD_DSYM(acGetMinNN,stream)
	LOAD_DSYM(acGetMaxNN,stream)
	LOAD_DSYM(acGetGridMaxNN,stream)
	LOAD_DSYM(acGetLengths,stream)
	LOAD_DSYM(acHostMeshCopyVertexBuffers,stream)
#include "device_load_uniform_loads.h"
	LOAD_DSYM(acHostMeshCopy,stream)
	LOAD_DSYM(acGetKernelId,stream)
	LOAD_DSYM(acGetKernelIdByName,stream)
	LOAD_DSYM(acCheckDeviceAvailability,stream)
	LOAD_DSYM(acGetNumDevicesPerNode,stream)
	LOAD_DSYM(acGetNumFields,stream)
	LOAD_DSYM(acGetFieldHandle,stream)
	LOAD_DSYM(acGetNode,stream)
	LOAD_DSYM(acNodeCreate,stream)
	LOAD_DSYM(acNodeDestroy,stream)
	LOAD_DSYM(acNodePrintInfo,stream)
	LOAD_DSYM(acNodeQueryDeviceConfiguration,stream)
	LOAD_DSYM(acNodeAutoOptimize,stream)
	LOAD_DSYM(acNodeSynchronizeStream,stream)
	LOAD_DSYM(acNodeSynchronizeVertexBuffer,stream)
	LOAD_DSYM(acNodeSynchronizeMesh,stream)
	LOAD_DSYM(acNodeSwapBuffers,stream)
	LOAD_DSYM(acNodeLoadConstant,stream)
	LOAD_DSYM(acNodeLoadVertexBufferWithOffset,stream)
	LOAD_DSYM(acNodeLoadMeshWithOffset,stream)
	LOAD_DSYM(acNodeLoadVertexBuffer,stream)
	LOAD_DSYM(acNodeLoadMesh,stream)
	LOAD_DSYM(acNodeSetVertexBuffer,stream)
	LOAD_DSYM(acNodeStoreVertexBufferWithOffset,stream)
	LOAD_DSYM(acNodeStoreMeshWithOffset,stream)
	LOAD_DSYM(acNodeStoreVertexBuffer,stream)
	LOAD_DSYM(acNodeStoreMesh,stream)
	LOAD_DSYM(acNodeIntegrateSubstep,stream)
	LOAD_DSYM(acNodeIntegrate,stream)
	LOAD_DSYM(acNodeIntegrateGBC,stream)
	LOAD_DSYM(acNodePeriodicBoundcondStep,stream)
	LOAD_DSYM(acNodePeriodicBoundconds,stream)
	LOAD_DSYM(acNodeGeneralBoundcondStep,stream)
	LOAD_DSYM(acNodeGeneralBoundconds,stream)
	LOAD_DSYM(acNodeReduceScal,stream)
	LOAD_DSYM(acNodeReduceVec,stream)
	LOAD_DSYM(acNodeReduceVecScal,stream)
	LOAD_DSYM(acDeviceCreate,stream)
	LOAD_DSYM(acDeviceDestroy,stream)
	LOAD_DSYM(acDeviceResetMesh,stream)
	LOAD_DSYM(acDevicePrintInfo,stream)
	LOAD_DSYM(acDeviceSynchronizeStream,stream)
	LOAD_DSYM(acDeviceSwapBuffer,stream)
	LOAD_DSYM(acDeviceSwapBuffers,stream)
	LOAD_DSYM(acDeviceLoadScalarUniform,stream)
	LOAD_DSYM(acDeviceLoadVectorUniform,stream)
	LOAD_DSYM(acDeviceLoadIntUniform,stream)
	LOAD_DSYM(acDeviceLoadBoolUniform,stream)
	LOAD_DSYM(acDeviceLoadInt3Uniform,stream)
	LOAD_DSYM(acDeviceStoreScalarUniform,stream)
	LOAD_DSYM(acDeviceStoreVectorUniform,stream)
	LOAD_DSYM(acDeviceStoreIntUniform,stream)
	LOAD_DSYM(acDeviceStoreBoolUniform,stream)
	LOAD_DSYM(acDeviceStoreInt3Uniform,stream)
	LOAD_DSYM(acDeviceLoadMeshInfo,stream)
	LOAD_DSYM(acDeviceLoadVertexBufferWithOffset,stream)
	LOAD_DSYM(acDeviceLoadMeshWithOffset,stream)
	LOAD_DSYM(acDeviceLoadVertexBuffer,stream)
	LOAD_DSYM(acDeviceLoadMesh,stream)
	LOAD_DSYM(acDeviceSetVertexBuffer,stream)
	LOAD_DSYM(acDeviceFlushOutputBuffers,stream)
	LOAD_DSYM(acDeviceStoreVertexBufferWithOffset,stream)
	LOAD_DSYM(acDeviceGetConfig,stream)
	LOAD_DSYM(acDeviceGetKernelInputParamsObject,stream)
	LOAD_DSYM(acDeviceStoreMeshWithOffset,stream)
	LOAD_DSYM(acDeviceStoreVertexBuffer,stream)
	LOAD_DSYM(acDeviceStoreMesh,stream)
	LOAD_DSYM(acDeviceTransferVertexBufferWithOffset,stream)
	LOAD_DSYM(acDeviceTransferMeshWithOffset,stream)
	LOAD_DSYM(acDeviceTransferVertexBuffer,stream)
	LOAD_DSYM(acDeviceTransferMesh,stream)
	LOAD_DSYM(acDeviceIntegrateSubstep,stream)
	LOAD_DSYM(acDevicePeriodicBoundcondStep,stream)
	LOAD_DSYM(acDevicePeriodicBoundconds,stream)
	LOAD_DSYM(acDeviceGeneralBoundcondStep,stream)
	LOAD_DSYM(acDeviceGeneralBoundconds,stream)
	LOAD_DSYM(acDeviceReduceScalNoPostProcessing,stream)
	LOAD_DSYM(acDeviceReduceScal,stream)
	LOAD_DSYM(acDeviceReduceVecNoPostProcessing,stream)
	LOAD_DSYM(acDeviceReduceVec,stream)
	LOAD_DSYM(acDeviceReduceVecScalNoPostProcessing,stream)
	LOAD_DSYM(acDeviceReduceVecScal,stream)
	LOAD_DSYM(acDeviceUpdate,stream)
	LOAD_DSYM(acDeviceGetKernelOutput,stream)
	LOAD_DSYM(acDeviceLaunchKernel,stream)
	LOAD_DSYM(acDeviceBenchmarkKernel,stream)
	LOAD_DSYM(acDeviceLoadStencil,stream)
	LOAD_DSYM(acDeviceLoadStencils,stream)
	LOAD_DSYM(acDeviceLoadStencilsFromConfig,stream)
	LOAD_DSYM(acDeviceStencilAccessesBoundaries,stream)
	LOAD_DSYM(acDeviceStoreStencil,stream)
	LOAD_DSYM(acDeviceVolumeCopy,stream)
#include "device_set_input_loads.h"
#include "device_get_input_loads.h"
#include "device_get_output_loads.h"
#include "get_vtxbufs_loads.h"

	*(void**)(&acDeviceGetIntOutput) = dlsym(handle,"acDeviceGetIntOutput");
	*(void**)(&acDeviceGetRealInput) = dlsym(handle,"acDeviceGetRealInput");
	*(void**)(&acDeviceGetIntInput) = dlsym(handle,"acDeviceGetIntInput");
	*(void**)(&acDeviceGetRealOutput) = dlsym(handle,"acDeviceGetRealOutput");
	LOAD_DSYM(acHostMeshCreate,stream)
	LOAD_DSYM(acHostGridMeshCreate,stream)
	LOAD_DSYM(acHostMeshRandomize,stream);
	LOAD_DSYM(acHostGridMeshRandomize,stream);
	LOAD_DSYM(acHostMeshDestroy,stream);

	LOAD_DSYM(acVerifyCompatibility,stream);
	LOAD_DSYM(acStoreConfig,stream);
//#ifdef __cplusplus
//	return AcLibHandle(handle);
//#else
//	return handle;
//#endif
	const AcResult is_compatible = acVerifyCompatibility(sizeof(AcMesh), sizeof(AcMeshInfo), sizeof(AcCompInfo), NUM_REAL_PARAMS, NUM_INT_PARAMS, NUM_BOOL_PARAMS, NUM_REAL_ARRAYS, NUM_INT_ARRAYS, NUM_BOOL_ARRAYS);
	if (is_compatible == AC_FAILURE)
	{
		fprintf(stderr,"Library is not compatible\n");
		exit(EXIT_FAILURE);
	}
	return AC_SUCCESS;
  }
  static AcResult __attribute__((unused)) acCloseLibrary()
  {
	const int success_closing_ac_lib = (astarothLibHandle != NULL) ? dlclose(astarothLibHandle) : 0;
	if(success_closing_ac_lib) astarothLibHandle = NULL;

	const int success_closing_kernels_lib = (kernelsLibHandle != NULL) ? dlclose(kernelsLibHandle) : 0;
	if(success_closing_kernels_lib) kernelsLibHandle = NULL;

	const int success_closing_utils_lib = (utilsLibHandle != NULL) ? dlclose(utilsLibHandle) : 0;
	if(success_closing_utils_lib) utilsLibHandle = NULL;

	return  (success_closing_ac_lib || success_closing_kernels_lib || success_closing_utils_lib) == 0 ? AC_SUCCESS : AC_FAILURE;
  }
#endif

/** Inits the profile to cosine wave */
AcResult acHostInitProfileToCosineWave(const AcReal spacing, const long offset,
                                       const AcReal amplitude, const AcReal wavenumber,
                                       const size_t profile_count, AcReal* profile);

/** Inits the profile to sine wave */
AcResult acHostInitProfileToSineWave(const AcReal spacing, const long offset,
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

#include "ac_buffer.h"

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

#if AC_DOUBLE_PRECISION
static UNUSED AcResult
acDeviceFinishReduce(Device device, const Stream stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output)
{
	return acDeviceFinishReduceFloat(device,stream,result,kernel,reduce_op,output);
}
#endif

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

#if AC_DOUBLE_PRECISION
static UNUSED AcResult
acDeviceFinishReduce(Device device, const cudaStream_t stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output)
{
	return acDeviceFinishReduceFloatStream(device,stream,result,kernel,reduce_op,output);
}
#endif

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
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0 , NULL, 0, NULL, 0, NULL, 0, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::vector<Profile> profile_in, std::vector<Profile> profile_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), profile_in.data(), profile_in.size(), profile_out.data(), profile_out.size(), 
		    			       NULL, 0,
		    			       NULL, 0, NULL, 0,
		                               loader);
}

static inline AcTaskDefinition
acCompute(AcKernel kernel, std::vector<Field> fields_in, std::vector<Field> fields_out, std::vector<Profile> profile_in, std::vector<Profile> profile_reduce_out, 
		std::vector<Profile> profile_write_out,
		std::vector<KernelReduceOutput> reduce_outputs_in, std::vector<KernelReduceOutput> reduce_outputs_out, std::function<void(ParamLoadingInfo)> loader)
{
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields_in.data(), fields_in.size(), fields_out.data(), fields_out.size(), profile_in.data(), profile_in.size(), profile_reduce_out.data(), profile_reduce_out.size(), 
		    			       profile_write_out.data(), profile_write_out.size(),
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
    return BASE_FUNC_NAME(acComputeWithParams)(kernel, fields.data(), fields.size(), fields.data(), fields.size(), NULL, 0 , NULL, 0, NULL, 0, NULL, 0, NULL, 0, loader);
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
	  {
		  config[param] = val;
		  config.is_loaded[param] = true;
	  }
  }

#endif
#include <string.h>

  static AcCompInfo UNUSED acInitCompInfo()
  {
	  AcCompInfo res;
	  //TP: initially nothing is loaded and if they are not loaded their values 
	  //might as well be zero since then a default value is used for them
	  memset(&res,0,sizeof(res));
	  return res;
  }
  static AcMeshInfo UNUSED acInitInfo()
  {
	  AcMeshInfo res;
	  //TP: this is useful for the following reasons:
	  //All enums are initialized by default to the first enum value
	  //All array ptrs are initialized to nulls
	  //All booleans are initialized to false
	  //All booleans about whether values are loaded are false
	  memset(&res,0,sizeof(res));
    	  // memset reads the second parameter as a byte even though it says int in
          // the function declaration
	  //TP: for backwards compatibility set original datatypes to all ones as before
    	  memset(&res.int_params,     (uint8_t)0xFF, sizeof(res.int_params));
    	  memset(&res.real_params,    (uint8_t)0xFF, sizeof(res.real_params));
    	  memset(&res.int3_params,    (uint8_t)0xFF, sizeof(res.int3_params));
    	  memset(&res.real3_params,   (uint8_t)0xFF, sizeof(res.real3_params));
    	  memset(&res.complex_params, (uint8_t)0xFF, sizeof(res.complex_params));

#if AC_MPI_ENABLED
	  res.comm = (AcCommunicator*)malloc(sizeof(AcCommunicator));
	  res.comm->handle = MPI_COMM_NULL;
#endif
	  res.run_consts = acInitCompInfo();
	  res.int3_params[AC_thread_block_loop_factors] = (int3){1,1,1};
	  res.int3_params[AC_max_tpb_for_reduce_kernels] = (int3){-1,8,8};
  	  res.runtime_compilation_log_dst = "/dev/stderr";
	  return res;
  }
  static AcMesh UNUSED acInitMesh()
  {
	  AcMesh res;
	  res.info = acInitInfo();
	  return res;
  }
