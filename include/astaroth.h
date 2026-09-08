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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

int3
acDecompose(const uint64_t target, const AcMeshInfo info);
int3
acGetPid3D(const uint64_t pid, const int3 decomp, const AcMeshInfo info);
int
acGetPid(const int3 pid, const int3 decomp, const AcMeshInfo info);

#include "get_vtxbufs_declares.h"

#if AC_RUNTIME_COMPILATION

static AcResult __attribute__((unused))
acLoadLibrary(const AcMeshInfo info)
{
    static int counter = 0;
    AcLibHandle handle = astarothLibHandle = acLibLoadLibrary(&info, "src/core", "libastaroth_core", &counter);

    LOAD_DSYM(acDeviceGetVertexBufferPtrs);
    LOAD_DSYM(acDeviceGetLocalConfig);
    LOAD_DSYM(acDeviceFinishReduceInt);
    LOAD_DSYM(acDeviceMemGetInfo);
    LOAD_DSYM(acKernelFlushInt);
    LOAD_DSYM(acAnalysisGetKernelInfo);
    LOAD_DSYM(acAnalysisCheckForDSLErrors);
    LOAD_DSYM(acDeviceSwapAllProfileBuffers);
    LOAD_DSYM(acDeviceFFTR2C);
    LOAD_DSYM(acDeviceFFTC2R);
    LOAD_DSYM(acDeviceFFTR2Planar);
    LOAD_DSYM(acDeviceFFTR2PlanarBatched);
    LOAD_DSYM(acDeviceFFTR2HermitianPlanarBatched);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2R);
    LOAD_DSYM(acDeviceFFTR2PlanarXY);
    LOAD_DSYM(acDeviceFFTBackwardTransformPlanar2RXY);
#if AC_MPI_ENABLED
    *(void**)(&BASE_FUNC_NAME(acBoundaryCondition)) = dlsym(handle, "acBoundaryCondition");
    LOAD_DSYM(ac_MPI_Barrier);
    LOAD_DSYM(ac_MPI_Comm_rank);
    LOAD_DSYM(ac_MPI_Comm_size);
    LOAD_DSYM(ac_MPI_Finalize);
    LOAD_DSYM(ac_MPI_Init);
    LOAD_DSYM(ac_MPI_Init_thread);
    LOAD_DSYM(acGridInitialized);
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
#ifdef AC_INTEGRATION_ENABLED
    LOAD_DSYM(acGridIntegrate);
#endif
    LOAD_DSYM(acGridSwapBuffers);
    LOAD_DSYM(acGridHaloExchange);
    LOAD_DSYM(acGridPeriodicBoundconds);
    LOAD_DSYM(acGridReduceScal);
    LOAD_DSYM(acGridReduceVec);
    LOAD_DSYM(acGridReduceVecScal);
    LOAD_DSYM(acGridReduceXY);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronous);
    LOAD_DSYM(acGridDiskAccessLaunch);
    LOAD_DSYM(acGridWriteSlicesToDiskLaunch);
    LOAD_DSYM(acGridWriteSlicesToDiskSynchronous);
    LOAD_DSYM(acGridWriteSlicesToDiskCollectiveSynchronous);
    LOAD_DSYM(acGridWriteMeshToDiskLaunch);
    LOAD_DSYM(acGridDiskAccessSync);
    LOAD_DSYM(acGridReadVarfileToMesh);
    LOAD_DSYM(acHaloExchangeBoundary);
    LOAD_DSYM(acPeriodicRay);
    *(void**)(&BASE_FUNC_NAME(acRayUpdate))          = dlsym(handle, "acRayUpdate");
    *(void**)(&BASE_FUNC_NAME(acComputeWithParams))  = dlsym(handle, "acComputeWithParams");
    *(void**)(&BASE_FUNC_NAME(acCompute))            = dlsym(handle, "acCompute");
    *(void**)(&BASE_FUNC_NAME(acHaloExchange))       = dlsym(handle, "acHaloExchange");
    *(void**)(&BASE_FUNC_NAME(acScan))               = dlsym(handle, "acScan");
    *(void**)(&BASE_FUNC_NAME(acGridBuildTaskGraph)) = dlsym(handle, "acGridBuildTaskGraph");
    *(void**)(&BASE_FUNC_NAME(
        acGridBuildTaskGraphWithBounds)) = dlsym(handle, "acGridBuildTaskGraphWithBounds");
    LOAD_DSYM(acGridDestroyTaskGraph);
    LOAD_DSYM(acGridClearTaskGraphCache);
    LOAD_DSYM(acGetDSLTaskGraph);
    *(void**)(&BASE_FUNC_NAME(acGetOptimizedDSLTaskGraph)) = dlsym(handle,
                                                                   "acGetOptimizedDSLTaskGraph");
    LOAD_DSYM(acGetDSLTaskGraphWithBounds);
    LOAD_DSYM(acGetOptimizedDSLTaskGraphWithBounds);
    LOAD_DSYM(acGetComputeStepsBCs);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronousDistributed);
    LOAD_DSYM(acGridAccessMeshOnDiskSynchronousCollective);
    LOAD_DSYM(acGridGetDefaultTaskGraph);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsX);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsY);
    LOAD_DSYM(acGridTaskGraphHasPeriodicBoundcondsZ);
    LOAD_DSYM(acGridTaskGraphIsEmpty);
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
    LOAD_DSYM(acGetLocalNN);
    LOAD_DSYM(acGetLocalMM);
    LOAD_DSYM(acGetGridNN);
    LOAD_DSYM(acGetGridMM);
    LOAD_DSYM(acGetMinNN);
    LOAD_DSYM(acGetMaxNN);
    LOAD_DSYM(acGetGridMaxNN);
    LOAD_DSYM(acGetLengths);
#include "device_load_uniform_loads.h"
#include "ac_push_to_config_loads.h"
    LOAD_DSYM(acGetKernelId);
    LOAD_DSYM(acGetKernelIdByName);
    LOAD_DSYM(acCheckDeviceAvailability);
    LOAD_DSYM(acGetNumDevicesPerNode);
    LOAD_DSYM(acGetNumFields);
    LOAD_DSYM(acGetFieldHandle);
    LOAD_DSYM(acGetFieldName);
    LOAD_DSYM(acFieldIsAuxiliary);
    LOAD_DSYM(acGetNode);
    LOAD_DSYM(acNodeCreate);
    LOAD_DSYM(acNodeDestroy);
    LOAD_DSYM(acNodePrintInfo);
    LOAD_DSYM(acNodeQueryDeviceConfiguration);
    LOAD_DSYM(acNodeAutoOptimize);
    LOAD_DSYM(acNodeSynchronizeStream);
    LOAD_DSYM(acNodeSynchronizeVertexBuffer);
    LOAD_DSYM(acNodeSynchronizeMesh);
    LOAD_DSYM(acNodeSwapBuffers);
    LOAD_DSYM(acNodeLoadConstant);
    LOAD_DSYM(acNodeLoadVertexBufferWithOffset);
    LOAD_DSYM(acNodeLoadMeshWithOffset);
    LOAD_DSYM(acNodeLoadVertexBuffer);
    LOAD_DSYM(acNodeLoadMesh);
    LOAD_DSYM(acNodeSetVertexBuffer);
    LOAD_DSYM(acNodeStoreVertexBufferWithOffset);
    LOAD_DSYM(acNodeStoreMeshWithOffset);
    LOAD_DSYM(acNodeStoreVertexBuffer);
    LOAD_DSYM(acNodeStoreMesh);
    LOAD_DSYM(acNodeIntegrateSubstep);
    LOAD_DSYM(acNodeIntegrate);
    LOAD_DSYM(acNodeIntegrateGBC);
    LOAD_DSYM(acNodePeriodicBoundcondStep);
    LOAD_DSYM(acNodePeriodicBoundconds);
    LOAD_DSYM(acNodeGeneralBoundcondStep);
    LOAD_DSYM(acNodeGeneralBoundconds);
    LOAD_DSYM(acNodeReduceScal);
    LOAD_DSYM(acNodeReduceVec);
    LOAD_DSYM(acNodeReduceVecScal);
    LOAD_DSYM(acDeviceCreate);
    LOAD_DSYM(acDeviceDestroy);
    LOAD_DSYM(acDeviceResetMesh);
    LOAD_DSYM(acDevicePrintInfo);
    LOAD_DSYM(acDeviceSynchronizeStream);
    LOAD_DSYM(acDeviceSwapBuffer);
    LOAD_DSYM(acDeviceSwapBuffers);
    LOAD_DSYM(acDeviceLoadScalarUniform);
    LOAD_DSYM(acDeviceLoadVectorUniform);
    LOAD_DSYM(acDeviceLoadIntUniform);
    LOAD_DSYM(acDeviceLoadBoolUniform);
    LOAD_DSYM(acDeviceLoadInt3Uniform);
    LOAD_DSYM(acDeviceStoreScalarUniform);
    LOAD_DSYM(acDeviceStoreVectorUniform);
    LOAD_DSYM(acDeviceStoreIntUniform);
    LOAD_DSYM(acDeviceStoreBoolUniform);
    LOAD_DSYM(acDeviceStoreInt3Uniform);
    LOAD_DSYM(acDeviceLoadMeshInfo);
    LOAD_DSYM(acDeviceLoadVertexBufferWithOffset);
    LOAD_DSYM(acDeviceLoadMeshWithOffset);
    LOAD_DSYM(acDeviceLoadVertexBuffer);
    LOAD_DSYM(acDeviceLoadMesh);
    LOAD_DSYM(acDeviceSetVertexBuffer);
    LOAD_DSYM(acDeviceFlushOutputBuffers);
    LOAD_DSYM(acDeviceStoreVertexBufferWithOffset);
    LOAD_DSYM(acDeviceGetConfig);
    LOAD_DSYM(acDeviceGetKernelInputParamsObject);
    LOAD_DSYM(acDeviceStoreMeshWithOffset);
    LOAD_DSYM(acDeviceStoreVertexBuffer);
    LOAD_DSYM(acDeviceStoreMesh);
    LOAD_DSYM(acDeviceTransferVertexBufferWithOffset);
    LOAD_DSYM(acDeviceTransferMeshWithOffset);
    LOAD_DSYM(acDeviceTransferVertexBuffer);
    LOAD_DSYM(acDeviceTransferMesh);
    LOAD_DSYM(acDeviceIntegrateSubstep);
    LOAD_DSYM(acDevicePeriodicBoundcondStep);
    LOAD_DSYM(acDevicePeriodicBoundconds);
    LOAD_DSYM(acDeviceGeneralBoundcondStep);
    LOAD_DSYM(acDeviceGeneralBoundconds);
    LOAD_DSYM(acDeviceReduceScalNoPostProcessing);
    LOAD_DSYM(acDeviceReduceScal);
    LOAD_DSYM(acDeviceReduceVecNoPostProcessing);
    LOAD_DSYM(acDeviceReduceVec);
    LOAD_DSYM(acDeviceReduceVecScalNoPostProcessing);
    LOAD_DSYM(acDeviceReduceVecScal);
    LOAD_DSYM(acDeviceUpdate);
    LOAD_DSYM(acDeviceGetKernelOutput);
    LOAD_DSYM(acDeviceLaunchKernel);
    LOAD_DSYM(acDeviceBenchmarkKernel);
    LOAD_DSYM(acDeviceLoadStencil);
    LOAD_DSYM(acDeviceLoadStencils);
    LOAD_DSYM(acDeviceLoadStencilsFromConfig);
    LOAD_DSYM(acDeviceStoreStencil);
    LOAD_DSYM(acDeviceVolumeCopy);
#include "device_get_input_loads.h"
#include "device_get_output_loads.h"
#include "device_set_input_loads.h"
#include "get_vtxbufs_loads.h"

    *(void**)(&acDeviceGetIntOutput)  = dlsym(handle, "acDeviceGetIntOutput");
    *(void**)(&acDeviceGetRealInput)  = dlsym(handle, "acDeviceGetRealInput");
    *(void**)(&acDeviceGetIntInput)   = dlsym(handle, "acDeviceGetIntInput");
    *(void**)(&acDeviceGetRealOutput) = dlsym(handle, "acDeviceGetRealOutput");
    LOAD_DSYM(acHostMeshRandomize);
    LOAD_DSYM(acHostGridMeshRandomize);
    LOAD_DSYM(acHostMeshDestroy);

    LOAD_DSYM(acVerifyCompatibility);
    LOAD_DSYM(acDeviceLoadRealReduceRes);
    // Runtime functions
    LOAD_DSYM(acKernelFlush);
    LOAD_DSYM(acVBAReset);
    LOAD_DSYM(acVBACreate);
    LOAD_DSYM(acAllocateArrays);
    LOAD_DSYM(acUpdateArrays);
    LOAD_DSYM(acVBADestroy);
    LOAD_DSYM(acRandInitAlt);
    LOAD_DSYM(acRandQuit);
    LOAD_DSYM(acLaunchKernel);
    LOAD_DSYM(acBenchmarkKernel);
    LOAD_DSYM(acLoadStencil);
    LOAD_DSYM(acStoreStencil);
    LOAD_DSYM(acLoadRealUniform);
    LOAD_DSYM(acLoadRealArrayUniform);
    LOAD_DSYM(acLoadReal3Uniform);
    LOAD_DSYM(acLoadIntUniform);
    LOAD_DSYM(acLoadIntUniform);
    LOAD_DSYM(acLoadIntArrayUniform);
    LOAD_DSYM(acLoadBoolUniform);
    LOAD_DSYM(acLoadIntArrayUniform);
    LOAD_DSYM(acLoadInt3Uniform);
    LOAD_DSYM(acStoreRealUniform);
    LOAD_DSYM(acStoreReal3Uniform);
    LOAD_DSYM(acStoreIntUniform);
    LOAD_DSYM(acStoreBoolUniform);
    LOAD_DSYM(acStoreInt3Uniform);
    LOAD_DSYM(acKernelLaunchGetLastTPB);
    LOAD_DSYM(acGetOptimizedKernel);
    LOAD_DSYM(acGetKernelReduceScratchPadSize);
    LOAD_DSYM(acGetKernelReduceScratchPadMinSize);
    LOAD_DSYM(acGetKernels);
    LOAD_DSYM(acGetOptimTPB);
    LOAD_DSYM(acRuntimeQuit);
    LOAD_DSYM(acGetRealScratchpadSize);

    const AcResult is_compatible = acVerifyCompatibility(sizeof(AcMesh), sizeof(AcMeshInfo),
                                                         sizeof(AcCompInfo), NUM_REAL_PARAMS,
                                                         NUM_INT_PARAMS, NUM_BOOL_PARAMS,
                                                         NUM_REAL_ARRAYS, NUM_INT_ARRAYS,
                                                         NUM_BOOL_ARRAYS);
    if (is_compatible == AC_FAILURE) {
        fprintf(stderr, "Library is not compatible\n");
        exit(EXIT_FAILURE);
    }
    return AC_SUCCESS;
}
static AcResult __attribute__((unused))
acCloseLibrary()
{
    const int success_closing_ac_lib = (astarothLibHandle != NULL) ? dlclose(astarothLibHandle) : 0;
    if (success_closing_ac_lib) astarothLibHandle = NULL;

    const int success_closing_utils_lib = (utilsLibHandle != NULL) ? dlclose(utilsLibHandle) : 0;
    if (success_closing_utils_lib) utilsLibHandle = NULL;

    return (success_closing_ac_lib || success_closing_utils_lib) == 0 ? AC_SUCCESS : AC_FAILURE;
}

#else

static AcResult __attribute__((unused))
acLoadLibrary(const AcMeshInfo info)
{
    return AC_FAILURE;
}

static AcResult __attribute__((unused))
acCloseLibrary()
{
    return AC_FAILURE;
}

#endif

static UNUSED AcCompInfo
acInitCompInfo()
{
    AcCompInfo res;
    // TP: initially nothing is loaded and if they are not loaded their values
    // might as well be zero since then a default value is used for them
    memset(&res, 0, sizeof(res));
    return res;
}

static UNUSED AcMeshInfo
acInitInfo()
{
    AcMeshInfo res;
    // TP: this is useful for the following reasons:
    // All enums are initialized by default to the first enum value
    // All array ptrs are initialized to nulls
    // All booleans are initialized to false
    // All booleans about whether values are loaded are false
    memset(&res, 0, sizeof(res));
    // memset reads the second parameter as a byte even though it says int in
    // the function declaration
    // TP: for backwards compatibility set original datatypes to all ones as before
    memset(&res.int_params, (uint8_t)0xFF, sizeof(res.int_params));
    memset(&res.real_params, (uint8_t)0xFF, sizeof(res.real_params));
    memset(&res.int3_params, (uint8_t)0xFF, sizeof(res.int3_params));
    memset(&res.real3_params, (uint8_t)0xFF, sizeof(res.real3_params));
    memset(&res.complex_params, (uint8_t)0xFF, sizeof(res.complex_params));

#if AC_MPI_ENABLED
    res.comm         = (AcCommunicator*)malloc(sizeof(AcCommunicator));
    res.comm->handle = MPI_COMM_NULL;
#endif
    res.run_consts = acInitCompInfo();
    return res;
}

static UNUSED AcMesh
acInitMesh()
{
    AcMesh res;
    for (size_t j = 0; j < NUM_VTXBUF_HANDLES; ++j) {
        res.vertex_buffer[j] = NULL;
    }
    res.info = acInitInfo();
    return res;
}

#include "ac_push_to_config_decl.h"

AC_END_C_DECLARATIONS

#ifdef __cplusplus

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
