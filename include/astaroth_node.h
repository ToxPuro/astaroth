#pragma once
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

