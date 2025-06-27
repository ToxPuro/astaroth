#pragma once
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
FUNC_DEFINE(AcResult, acDeviceFFTR2C,(const Device device, const Field src, const ComplexField dst));
FUNC_DEFINE(AcResult, acDeviceFFTC2R,(const Device device, const ComplexField src, const Field dst));

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
                                        Device dst_device));

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
                                       const VertexBufferHandle vtxbuf_handle, const Volume start,
                                       const Volume end));

/** */
FUNC_DEFINE(AcResult, acDevicePeriodicBoundconds,(const Device device, const Stream stream, const Volume start,
                                    const Volume end));

/** */
FUNC_DEFINE(AcResult, acDeviceGeneralBoundcondStep,(const Device device, const Stream stream,
                                      const VertexBufferHandle vtxbuf_handle, const Volume start,
                                      const Volume end, const AcMeshInfo config, const int3 bindex));

/** */
FUNC_DEFINE(AcResult, acDeviceGeneralBoundconds,(const Device device, const Stream stream, const Volume start,
                                   const Volume end, const AcMeshInfo config, const int3 bindex));

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
FUNC_DEFINE(AcResult, acDeviceReduceXYAverages,(const Device , const Stream));

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
#if AC_DOUBLE_PRECISION
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceFloat,(Device device, const Stream stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output));
#endif
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceInt,(Device device, const Stream stream, int* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcIntOutputParam output));
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceRealStream,(Device device, const cudaStream_t stream, AcReal* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcRealOutputParam output));
#if AC_DOUBLE_PRECISION
/** */
FUNC_DEFINE(AcResult,  acDeviceFinishReduceFloatStream,(Device device, const cudaStream_t stream, float* result,const AcKernel kernel, const AcReduceOp reduce_op, const AcFloatOutputParam output));
#endif
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
FUNC_DEFINE(AcResult, acDeviceSetReduceOffset,(const Device device, const AcKernel kernel,
                              const Volume start, const Volume end));

/** */
FUNC_DEFINE(AcResult, acDeviceBenchmarkKernel,(const Device device, const AcKernel kernel, const int3 start,
                                 const int3 end));
/** */
FUNC_DEFINE(AcResult, acDeviceLoadStencilsFromConfig,(const Device device, const Stream stream));

/** */
FUNC_DEFINE(AcBoundary, acDeviceStencilAccessesBoundaries,(const Device device, const Stencil stencil));

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
FUNC_DEFINE(AcResult, acDeviceMemGetInfo,(const Device device, size_t* free_mem, size_t* total_mem));

/** */
AcResult acDeviceWriteMeshToDisk(const Device device, const VertexBufferHandle vtxbuf,
                                 const char* filepath);
