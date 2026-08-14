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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ac_helpers.h"
#include "ac_mpi.h"
#include "acc_runtime.h"
#include "acreal.h"
#include "astaroth_analysis.h"
#include "astaroth_base.h"
#include "astaroth_device.h"
// The device headers are not needed for interfacing with Astaroth
#include "astaroth_device_headers.h"
#include "astaroth_grid.h"
#include "astaroth_helpers.h"
#include "astaroth_legacy.h"
#include "astaroth_logging.h"
#include "astaroth_node.h"
#include "astaroth_runtime_compilation.h"
#include "builtin_enums.h"
#include "errchk.h"
#include "func_define.h"
#include "host_datatypes.h"

// clang-format off
#include "user_defines.h"
// clang-format on

#if AC_RUNTIME_COMPILATION
#include "astaroth_lib.h"
#endif

AC_BEGIN_C_DECLARATIONS

FUNC_DEFINE(Node, acGetNode,(void));

int3
acDecompose(const uint64_t target, const AcMeshInfo info);
int3
acGetPid3D(const uint64_t pid, const int3 decomp, const AcMeshInfo info);
int
acGetPid(const int3 pid, const int3 decomp, const AcMeshInfo info);

#include "get_vtxbufs_declares.h"

#if AC_RUNTIME_COMPILATION
#define LOAD_DSYM(FUNC_NAME,STREAM) *(void**)(&FUNC_NAME) = dlsym(handle,#FUNC_NAME); \
			     if(!FUNC_NAME && STREAM) fprintf(STREAM,"Astaroth warning: was not able to load %s\n",#FUNC_NAME);

  static AcResult __attribute__((unused)) acLoadLibrary(FILE* stream, const AcMeshInfo info)
  {
	const size_t len = 20000;
	char original_runtime_astaroth_path[len];
#ifdef __APPLE__
	snprintf(original_runtime_astaroth_path,len,"%s/runtime_build/src/core/libastaroth_core.dylib",info.runtime_compilation_build_path ? info.runtime_compilation_build_path : astaroth_binary_path);
#else
	snprintf(original_runtime_astaroth_path,len,"%s/runtime_build/src/core/libastaroth_core.so",info.runtime_compilation_build_path ? info.runtime_compilation_build_path : astaroth_binary_path);
#endif
	static int counter = 0;
	const char* runtime_astaroth_path = acLibraryVersion(original_runtime_astaroth_path,counter,info.comm);
	++counter;
 	void* handle = dlopen(runtime_astaroth_path,RTLD_NOW | RTLD_LOCAL);
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
	LOAD_DSYM(acDeviceMemGetInfo,stream)
	LOAD_DSYM(acKernelFlushInt,stream) 
	LOAD_DSYM(acAnalysisGetKernelInfo,stream)
	LOAD_DSYM(acAnalysisCheckForDSLErrors,stream)
        LOAD_DSYM(acDeviceSwapAllProfileBuffers,stream)
	LOAD_DSYM(acDeviceFFTR2C,stream)
	LOAD_DSYM(acDeviceFFTC2R,stream)
	LOAD_DSYM(acDeviceFFTR2Planar,stream)
	LOAD_DSYM(acDeviceFFTR2PlanarBatched,stream)
	LOAD_DSYM(acDeviceFFTR2HermitianPlanarBatched,stream)
	LOAD_DSYM(acDeviceFFTBackwardTransformPlanar,stream)
	LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2R,stream)
	LOAD_DSYM(acDeviceFFTR2PlanarXY,stream)
	LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2RXY,stream)
#if AC_MPI_ENABLED
	*(void**)(&BASE_FUNC_NAME(acBoundaryCondition)) = dlsym(handle,"acBoundaryCondition");
	LOAD_DSYM(ac_MPI_Init,stream)
	LOAD_DSYM(ac_MPI_Init_thread,stream)
	LOAD_DSYM(ac_MPI_Finalize,stream);
	LOAD_DSYM(ac_MPI_Comm_rank,stream);
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
#ifdef AC_INTEGRATION_ENABLED
	LOAD_DSYM(acGridIntegrate,stream);
#endif
	LOAD_DSYM(acGridSwapBuffers,stream);
	LOAD_DSYM(acGridHaloExchange,stream);
	LOAD_DSYM(acGridPeriodicBoundconds,stream);
	LOAD_DSYM(acGridReduceScal,stream);
	LOAD_DSYM(acGridReduceVec,stream);
	LOAD_DSYM(acGridReduceVecScal,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronous,stream);
	LOAD_DSYM(acGridDiskAccessLaunch,stream);
	LOAD_DSYM(acGridWriteSlicesToDiskLaunch,stream);
	LOAD_DSYM(acGridWriteSlicesToDiskSynchronous,stream);
	LOAD_DSYM(acGridWriteSlicesToDiskCollectiveSynchronous,stream);
	LOAD_DSYM(acGridWriteMeshToDiskLaunch,stream);
	LOAD_DSYM(acGridDiskAccessSync,stream);
	LOAD_DSYM(acGridReadVarfileToMesh,stream);
	LOAD_DSYM(acHaloExchangeBoundary,stream);
	LOAD_DSYM(acPeriodicRay,stream);
	*(void**)(&BASE_FUNC_NAME(acRayUpdate)) = dlsym(handle,"acRayUpdate");
	*(void**)(&BASE_FUNC_NAME(acComputeWithParams)) = dlsym(handle,"acComputeWithParams");
	*(void**)(&BASE_FUNC_NAME(acCompute)) = dlsym(handle,"acCompute");
	*(void**)(&BASE_FUNC_NAME(acHaloExchange)) = dlsym(handle,"acHaloExchange");
	*(void**)(&BASE_FUNC_NAME(acScan)) = dlsym(handle,"acScan");
	*(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle,"acGridBuildTaskGraph");
	*(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraphWithBounds)) = dlsym(handle,"acGridBuildTaskGraphWithBounds");
	LOAD_DSYM(acGridDestroyTaskGraph,stream);
	LOAD_DSYM(acGridClearTaskGraphCache,stream);
	LOAD_DSYM(acGetDSLTaskGraph,stream);
	*(void**)(&BASE_FUNC_NAME(acGetOptimizedDSLTaskGraph)) = dlsym(handle,"acGetOptimizedDSLTaskGraph");
	LOAD_DSYM(acGetDSLTaskGraphWithBounds,stream);
	LOAD_DSYM(acGetOptimizedDSLTaskGraphWithBounds,stream);
	LOAD_DSYM(acGetComputeStepsBCs,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousDistributed,stream);
	LOAD_DSYM(acGridAccessMeshOnDiskSynchronousCollective,stream);
	LOAD_DSYM(acGridGetDefaultTaskGraph,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsX,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsY,stream);
	LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsZ,stream);
	LOAD_DSYM(acGridTaskGraphIsEmpty,stream);
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
	LOAD_DSYM(acGetFieldName,stream)
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
	LOAD_DSYM(acDeviceLoadRealReduceRes,stream);
	//Runtime functions
	LOAD_DSYM(acKernelFlush,stream);
	LOAD_DSYM(acVBAReset,stream);
	LOAD_DSYM(acVBACreate,stream);
	LOAD_DSYM(acAllocateArrays,stream);
	LOAD_DSYM(acUpdateArrays,stream);
	LOAD_DSYM(acVBADestroy,stream);
	LOAD_DSYM(acRandInitAlt,stream);
	LOAD_DSYM(acRandQuit,stream);
	LOAD_DSYM(acLaunchKernel,stream);
	LOAD_DSYM(acBenchmarkKernel,stream);
	LOAD_DSYM(acLoadStencil,stream);
	LOAD_DSYM(acStoreStencil,stream);
	LOAD_DSYM(acLoadRealUniform,stream);
	LOAD_DSYM(acLoadRealArrayUniform,stream);
	LOAD_DSYM(acLoadReal3Uniform,stream);
	LOAD_DSYM(acLoadIntUniform,stream)
	LOAD_DSYM(acLoadIntUniform,stream)
	LOAD_DSYM(acLoadIntArrayUniform,stream)
	LOAD_DSYM(acLoadBoolUniform,stream)
	LOAD_DSYM(acLoadIntArrayUniform,stream)
	LOAD_DSYM(acLoadInt3Uniform,stream)
	LOAD_DSYM(acStoreRealUniform,stream)
	LOAD_DSYM(acStoreReal3Uniform,stream)
	LOAD_DSYM(acStoreIntUniform,stream)
	LOAD_DSYM(acStoreBoolUniform,stream)
	LOAD_DSYM(acStoreInt3Uniform,stream)
	LOAD_DSYM(acKernelLaunchGetLastTPB,stream)
	LOAD_DSYM(acGetOptimizedKernel,stream)
	LOAD_DSYM(acGetKernelReduceScratchPadSize,stream)
	LOAD_DSYM(acGetKernelReduceScratchPadMinSize,stream)
	LOAD_DSYM(acGetKernels,stream)
	LOAD_DSYM(acGetOptimTPB,stream);
        LOAD_DSYM(acRuntimeQuit,stream);
	LOAD_DSYM(acGetRealScratchpadSize,stream);
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

	const int success_closing_utils_lib = (utilsLibHandle != NULL) ? dlclose(utilsLibHandle) : 0;
	if(success_closing_utils_lib) utilsLibHandle = NULL;

	return  (success_closing_ac_lib || success_closing_utils_lib) == 0 ? AC_SUCCESS : AC_FAILURE;
  }
#else
  static AcResult __attribute__((unused)) acLoadLibrary(FILE*, const AcMeshInfo) {return AC_FAILURE;}
  static AcResult __attribute__((unused)) acCloseLibrary() {return AC_FAILURE;}
#endif

AC_END_C_DECLARATIONS

#ifdef __cplusplus

static inline AcReal*
acHostCreateVertexBuffer(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acHostCreateVertexBufferVariable(info,vtxbuf);
}

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
   const int3 halos = acGetFieldHalos(info,vtxbuf);
   const Volume n0 = 
          (Volume)
          {
                  as_size_t(halos.x),
                  as_size_t(halos.y),
                  as_size_t(halos.z)
          };
   const Volume m1 = 
	   (Volume){
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].x),
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].y),
		as_size_t(info.int3_params[vtxbuf_dims[vtxbuf]].z)
	   };
   const Volume n1 = 
	   (Volume)
	   {
	   	m1.x-n0.x,
	   	m1.y-n0.y,
	   	m1.z-n0.z,
	   };
   const Volume m0 = (Volume){0, 0, 0};
   const Volume nn = 
	   (Volume)
	   {
	   	m1.x-2*n0.x,
	   	m1.y-2*n0.y,
	   	m1.z-2*n0.z,
	   };
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

static inline size_t
acVertexBufferIdx(const int i, const int j, const int k, const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acVertexBufferIdxVariable(i,j,k,info,vtxbuf);
}

static inline Volume 
acVertexBufferDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
	return acVertexBufferDimsVariable(info,vtxbuf);
}

static inline size_t
acVertexBufferSizeBytes(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
    return acVertexBufferSizeBytesVariable(info,vtxbuf);
}

static inline size_t
acVertexBufferCompdomainSize(const AcMeshInfo info, const VertexBufferHandle vtxbuf) {return acVertexBufferCompdomainSizeVariable(info,vtxbuf);}

static inline size_t
acVertexBufferCompdomainSizeBytes(const AcMeshInfo info, const VertexBufferHandle vtxbuf) {return  acVertexBufferCompdomainSizeBytesVariable(info,vtxbuf); }

template <typename P, typename V>
void
acPushToConfig(AcMeshInfo& config, P param, V val)
{
	(void)param;
	(void)val;
        if constexpr(IsCompParam<P>())
        {
        	  config.run_consts.config[param] = val;
        	  config.run_consts.is_loaded[param] = true;
        }
        else if constexpr(IsParam<P>())
        {
      	  config[param] = val;
      	  config.is_loaded[param] = true;
        }
}

static AcResult
acUpdateDecompositionParams(AcMeshInfo* dst)
{
#if AC_MPI_ENABLED
	int nprocs{};
	int rank{};
	ERRCHK_ALWAYS(dst->comm != NULL && dst->comm->handle != MPI_COMM_NULL);
	MPI_Comm_size(dst->comm->handle,&nprocs);
	MPI_Comm_rank(dst->comm->handle,&rank);
	const int3 decomp = acDecompose(nprocs,*dst);
	const int3 pid3d = acGetPid3D(rank,decomp,*dst);
	acPushToConfig((*dst),AC_domain_coordinates,pid3d);
	acPushToConfig((*dst),AC_domain_decomposition,decomp);
	return AC_SUCCESS;
#else
	return AC_FAILURE;
#endif
}

#endif

  static UNUSED AcCompInfo acInitCompInfo()
  {
	  AcCompInfo res;
	  //TP: initially nothing is loaded and if they are not loaded their values 
	  //might as well be zero since then a default value is used for them
	  memset(&res,0,sizeof(res));
	  return res;
  }
  static UNUSED AcMeshInfo acInitInfo()
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
	  return res;
  }
  static UNUSED AcMesh acInitMesh()
  {
	  AcMesh res;
	  for(size_t j = 0; j < NUM_VTXBUF_HANDLES; ++j)
	  {
		  res.vertex_buffer[j] = NULL;
	  }
	  res.info = acInitInfo();
	  return res;
  }
