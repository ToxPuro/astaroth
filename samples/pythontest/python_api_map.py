

#NOTE: This is a draft for the relevant classes etc. to plan the Python implementation of Astaroth. 
# DO NOT ATTEMPT TO RUN!!! 
# WORK IN PROGRESS: DO not attempt to implement before MV gives green light! 

# Grid interface

AcMeshInfo, Stream, AcRealParam, AcReal3Param, AcReal, AcReal3, AcMesh,
ReductionType, VertexBufferHandle, device, AcIntParam, AcInt3Param, int, int3,
ScalarArrayHandle, size_t #define

astaroth.grid.acGridInit(info)

astaroth.grid.acGridQuit()

astaroth.grid.acGridSynchronizeStream(stream)

astaroth.grid.acGridLoadScalarUniform(stream, param, value)

astaroth.grid.acGridLoadVectorUniform(stream, param, value)

astaroth.grid.acGridLoadMesh(stream, host_mesh)

host_mesh = astaroth.grid.acGridStoreMesh(stream)

astaroth.grid.acGridIntegrate(stream, dt)

astaroth.grid.acGridPeriodicBoundconds(stream)

result = astaroth.grid.acGridReduceScal(stream, rtype, vtxbuf_handle, result)
 
result = astaroth.grid.acGridReduceVec(stream,  rtype, vtxbuf0, vtxbuf1, vtxbuf2, result)


 # Device interface

device = astaroth.device.acDeviceCreate(id, device_config)

astaroth.device.acDeviceDestroy(device)

astaroth.device.acDevicePrintInfo(device)

astaroth.device.acDeviceAutoOptimize(device)

astaroth.device.acDeviceSynchronizeStream(device, stream)

astaroth.device.acDeviceSwapBuffers(device)

astaroth.device.acDeviceLoadScalarUniform(device, stream, AcRealParam param,  AcReal value)

astaroth.device.acDeviceLoadVectorUniform(device, stream, AcReal3Param param, AcReal3 value)

astaroth.device.acDeviceLoadIntUniform( device, stream, param, value)

astaroth.device.acDeviceLoadInt3Uniform(device, stream, param, value)

data = astaroth.device.acDeviceLoadScalarArray(device, stream, ScalarArrayHandle handle, start, num)

astaroth.device.acDeviceLoadMeshInfo(device, device_config)

astaroth.device.acDeviceLoadDefaultUniforms(device)

astaroth.device.acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, vtxbuf_handle, src, dst, num_vertices)

astaroth.device.acDeviceLoadVertexBuffer(device, stream, host_mesh, vtxbuf_handle)

astaroth.device.acDeviceLoadMesh(device, stream, host_mesh)

host_mesh = astaroth.device.acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src, dst, num_vertices)

host_mesh = astaroth.device.acDeviceStoreMeshWithOffset(device, stream, src, dst, num_vertices)

host_mesh = astaroth.device.acDeviceStoreVertexBuffer(device, stream, vtxbuf_handle, host_mesh)

host_mesh = astaroth.device.acDeviceStoreMesh(device, stream)

astaroth.device.acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst, num_vertices, dst_device)

astaroth.device.acDeviceTransferVertexBuffer( src_device, stream, vtxbuf_handle, dst_device)

astaroth.device.acDeviceTransferMesh(src_device, stream, dst_device)

astaroth.device.acDeviceIntegrateSubstep(device, stream, step_number, start, end, dt)

astaroth.device.acDevicePeriodicBoundcondStep(device, stream, vtxbuf_handle, start, end)

astaroth.device.acDevicePeriodicBoundconds(device, stream, start, end)

result = astaroth.device.acDeviceReduceScal(device, stream, rtype, vtxbuf_handle, result)

result = astaroth.device.acDeviceReduceVec(device, stream_type,     rtype, vtxbuf0, vtxbuf1, vtxbuf2)

result = astaroth.device.acDeviceReduceVecScal(device, stream_type, rtype, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, result)

# Helper functions

config = astaroth.helper.acUpdateBuiltinParams(config);
mesh   = astaroth.helper.acMeshCreate(mesh_info, mesh);
mesh   = astaroth.helper.acMeshDestroy(mesh);

