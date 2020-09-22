

#NOTE: This is a draft for the relevant classes etc. to plan the Python implementation of Astaroth. 
# DO NOT ATTEMPT TO RUN!!! 
# WORK IN PROGRESS: DO not attempt to implement before MV gives green light! 

# Grid interface

AcMeshInfo, Stream, AcRealParam, AcReal3Param, AcReal, AcReal3, AcMesh,
ReductionType, VertexBufferHandle, device, AcIntParam, AcInt3Param, int, int3,
ScalarArrayHandle, size_t #define

import aclib # Library of Astaroth functions


#Dratring class wrappter for Grid
class Grid:
    def __init__(self):
        #Somehow we need to interface Struct Grid and python. If we could call C datatypes as  
        self.device
        self.submesh
        self.decomposition
        self.initialized
        self.nn = (xx,xx,xx)
        self.corner_data
        self.edgex_data
        self.edgey_data
        self.edgez_data
        self.sidexy_data
        self.sidexz_data
        self.sideyz_data
        self.comm_cart


# Class wrapper for pythonic usability and higher lever wrapping on connected operations
class AcIntegrator:
    def __init__(self, pid, modelmesh):
        # Set mesh configurations
        self.info      = aclib.io.acLoadConfig(AC_DEFAULT_CONFIG)
        # Set the model mesh as part of the object
        self.modelmesh = modelmesh
        # GPU alloc & compute (MV: This thing I do not understand, Now dealing with grid creatively. )
        self.grid = Grid()
        self.grid = aclib.grid.acGridInit(self.info, self.grid) 

        # What is STREAM_DEFAULT? Can be just update grid implicitly - can it be seen as a global variable by python?  
        self.grid = aclib.grid.acGridLoadMesh(STREAM_DEFAULT, self.modelmesh, self.grid)

    def halo_exchange(self)
        self.grid = aclib.grid.acGridPeriodicBoundconds(STREAM_DEFAULT, self.grid)
        
    def quit(self)
        aclib.grid.acGridQuit()


aclib.grid.acGridSynchronizeStream(stream)

aclib.grid.acGridLoadScalarUniform(stream, param, value)

aclib.grid.acGridLoadVectorUniform(stream, param, value)

aclib.grid.acGridLoadMesh(stream, host_mesh)

host_mesh = aclib.grid.acGridStoreMesh(stream)

aclib.grid.acGridIntegrate(stream, dt)

aclib.grid.acGridPeriodicBoundconds(stream)

result = aclib.grid.acGridReduceScal(stream, rtype, vtxbuf_handle, result)
 
result = aclib.grid.acGridReduceVec(stream,  rtype, vtxbuf0, vtxbuf1, vtxbuf2, result)


 # Device interface

device = aclib.device.acDeviceCreate(id, device_config)

aclib.device.acDeviceDestroy(device)

aclib.device.acDevicePrintInfo(device)

aclib.device.acDeviceAutoOptimize(device)

aclib.device.acDeviceSynchronizeStream(device, stream)

aclib.device.acDeviceSwapBuffers(device)

aclib.device.acDeviceLoadScalarUniform(device, stream, AcRealParam param,  AcReal value)

aclib.device.acDeviceLoadVectorUniform(device, stream, AcReal3Param param, AcReal3 value)

aclib.device.acDeviceLoadIntUniform( device, stream, param, value)

aclib.device.acDeviceLoadInt3Uniform(device, stream, param, value)

data = aclib.device.acDeviceLoadScalarArray(device, stream, ScalarArrayHandle handle, start, num)

aclib.device.acDeviceLoadMeshInfo(device, device_config)

aclib.device.acDeviceLoadDefaultUniforms(device)

aclib.device.acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, vtxbuf_handle, src, dst, num_vertices)

aclib.device.acDeviceLoadVertexBuffer(device, stream, host_mesh, vtxbuf_handle)

aclib.device.acDeviceLoadMesh(device, stream, host_mesh)

host_mesh = aclib.device.acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src, dst, num_vertices)

host_mesh = aclib.device.acDeviceStoreMeshWithOffset(device, stream, src, dst, num_vertices)

host_mesh = aclib.device.acDeviceStoreVertexBuffer(device, stream, vtxbuf_handle, host_mesh)

host_mesh = aclib.device.acDeviceStoreMesh(device, stream)

aclib.device.acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst, num_vertices, dst_device)

aclib.device.acDeviceTransferVertexBuffer( src_device, stream, vtxbuf_handle, dst_device)

aclib.device.acDeviceTransferMesh(src_device, stream, dst_device)

aclib.device.acDeviceIntegrateSubstep(device, stream, step_number, start, end, dt)

aclib.device.acDevicePeriodicBoundcondStep(device, stream, vtxbuf_handle, start, end)

aclib.device.acDevicePeriodicBoundconds(device, stream, start, end)

result = aclib.device.acDeviceReduceScal(device, stream, rtype, vtxbuf_handle, result)

result = aclib.device.acDeviceReduceVec(device, stream_type,     rtype, vtxbuf0, vtxbuf1, vtxbuf2)

result = aclib.device.acDeviceReduceVecScal(device, stream_type, rtype, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3, result)

# Helper functions

config = aclib.helper.acUpdateBuiltinParams(config);
mesh   = aclib.helper.acMeshCreate(mesh_info, mesh);
mesh   = aclib.helper.acMeshDestroy(mesh);


#For reference to see everything in use. 
int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acMeshCreate(info, &model);
        acMeshCreate(info, &candidate);
        acMeshRandomize(&model);
        acMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // Boundconds
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Boundconds", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        acMeshRandomize(&model);
    }

    // Integration
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridIntegrate(STREAM_DEFAULT, FLT_EPSILON);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acModelIntegrateStep(model, FLT_EPSILON);
        acMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        ERRCHK_ALWAYS(res == AC_SUCCESS);
        acMeshRandomize(&model);
    }

    // Scalar reductions
    acGridLoadMesh(STREAM_DEFAULT, model);

    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, (ReductionType)i, v0, &candval);
        if (pid == 0) {
            const AcReal modelval   = acModelReduceScal(model, (ReductionType)i, v0);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acModelReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acModelReduceScal(model, RTYPE_MIN, v0);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
        printf("Warning: testing only RTYPE_MAX and RTYPE_MIN\n");
        fflush(stdout);
    }
    for (size_t i = 0; i < 2; ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = VTXBUF_UUX;
        const VertexBufferHandle v1 = VTXBUF_UUY;
        const VertexBufferHandle v2 = VTXBUF_UUZ;
        AcReal candval;
        acGridReduceVec(STREAM_DEFAULT, (ReductionType)i, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval   = acModelReduceVec(model, (ReductionType)i, v0, v1, v2);
            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acModelReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acModelReduceVec(model, RTYPE_MIN, v0, v1, v1);
            ERRCHK_ALWAYS(acEvalError(rtype_names[i], error));
        }
    }

    if (pid == 0) {
        acMeshDestroy(&model);
        acMeshDestroy(&candidate);
    }

    acGridQuit();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
