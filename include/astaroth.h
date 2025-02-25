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
    printf("%s: %d\n", intparam_names[a], info.params.scalars.int_params[a]);
}

void acPrintIntParams(const AcIntParam a, const AcIntParam b, const AcIntParam c,
                      const AcMeshInfo info);

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
 * Helper functions
 * =============================================================================
 */
AcResult 
acHostUpdateBuiltinParams(AcMeshInfo* config);

AcResult 
acHostUpdateBuiltinCompParams(AcMeshInfo* config);



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

/** Sets the dimensions of the computational grid to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */
FUNC_DEFINE(AcResult, acSetMeshDims,(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info));

/** Sets the dimensions of the computational subdomain to (nx, ny, nz) and recalculates the built-in
 * parameters derived from them (mx, my, mz, nx_min, and others) */

FUNC_DEFINE(AcResult, acSetSubMeshDims,(const size_t nx, const size_t ny, const size_t nz, AcMeshInfo* info));

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
	LOAD_DSYM(acSetMeshDims)
	LOAD_DSYM(acSetSubMeshDims)
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
	LOAD_DSYM(acGridMPISubComms);
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
	LOAD_DSYM(acDeviceStencilAccessesBoundaries)
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
AcResult acReduceProfileWithBounds(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream, const Volume start, const Volume end, const Volume start_after_transpose, const Volume end_after_transpose);

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
		  config[param] = val;
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
	  memset(&res,0,sizeof(res));
    	  // memset reads the second parameter as a byte even though it says int in
          // the function declaration
	  //TP: for backwards compatibility set original datatypes to all ones as before
    	  memset(&res.params.scalars.int_params,     (uint8_t)0xFF, sizeof(res.params.scalars.int_params));
    	  memset(&res.params.scalars.real_params,    (uint8_t)0xFF, sizeof(res.params.scalars.real_params));
    	  memset(&res.params.scalars.int3_params,    (uint8_t)0xFF, sizeof(res.params.scalars.int3_params));
    	  memset(&res.params.scalars.real3_params,   (uint8_t)0xFF, sizeof(res.params.scalars.real3_params));
    	  memset(&res.params.scalars.complex_params, (uint8_t)0xFF, sizeof(res.params.scalars.complex_params));

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
