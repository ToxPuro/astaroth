
import ctypes
import pathlib

libname = pathlib.Path().absolute() / "acc-runtime/api/libacc-runtime.a"
libacc  = ctypes.CDLL(libname)

#MV: In Python interace we cannot use any preprocessor macro stuff. Therefore
#MV: NUM_INT_PARAMS etc. need to be set manually for now.

#MV: Note mandatory double precision.

class py_int3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("z", ctypes.c_int)
               ]

class py_real3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("z", ctypes.c_double)
               ]

class py_vertex_buffer(ctypes.Structure):
    _fields_ = [(,)
               ]

'''
AcMeshInfo mesh_info
typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;
'''
class py_AcMeshInfo(ctypes.Structure):
    _fields_ = [("int_params",  NUM_INT_PARAMS   * ctypes.c_int),
                ("int_params",  NUM_INT3_PARAMS  * py_int3),
                ("real_params", NUM_REAL_PARAMS  * ctypes.c_double),
                ("rea3_params", NUM_REAL3_PARAMS * py_real3)
               ]
'''
typedef struct {
    AcReal* vertex_buffer[NUM_VTXBUF_HANDLES];
    AcMeshInfo info;
} AcMesh;
'''
class py_AcMesh(ctypes.Structure):
    _fields_ = [("vertex_buffer", NUM_VTXBUF_HANDLES * py_vertex_buffer),
                ("info", py_AcMeshInfo)
               ]

def py_load_config(config_path):
    ...
    return mesh_info

def py_acmesh_create(mesh_info):
    ...
    return mesh

def py_initialize_mesh("random_fields", mesh):
    ...
    return mesh


'''
=============================================================================
Device interface
=============================================================================
'''

def py_acDeviceCreate(id, device_config, device):
    #interface variables
    #const int id
    #const AcMeshInfo device_config
    #Device* device
    return py_AcResult

def py_acDeviceDestroy(device):
    #interface variables
    #Device device
    return py_AcResult

def py_acDevicePrintInfo(device):
    #interface variables
    #const Device device
    return py_AcResult

def py_acDeviceSynchronizeStream(device, stream):
    #interface variables
    #const Device device
    #const Stream stream
    return py_AcResult

def py_acDeviceSwapBuffer(device, handle):
    #interface variables
    #const Device device
    #const VertexBufferHandle handle
    return py_AcResult

def py_acDeviceSwapBuffers(device):
    #interface variables
    #const Device device
    return py_AcResult

def py_acDeviceLoadScalarUniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcRealParam param
    #const AcReal value
    return py_AcResult

def py_acDeviceLoadVectorUniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream,
    #const AcReal3Param param
    #const AcReal3 value
    return py_AcResult

def py_acDeviceLoadIntUniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcIntParam param
    #const int value
    return py_AcResult

def py_acDeviceLoadInt3Uniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcInt3Param param
    #const int3 value
    return py_AcResult

def py_acDeviceStoreScalarUniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream,
    #const AcRealParam param
    #AcReal* value
    return py_AcResult

def py_acDeviceStoreVectorUniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcReal3Param param
    #AcReal3* value
    return py_AcResult

def py_acDeviceStoreIntUniform(device, stream, param, value):
    #interface variables
    #Device device
    #const Stream stream
    #const AcIntParam param
    #int* value
    return py_AcResult

def py_acDeviceStoreInt3Uniform(device, stream, param, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcInt3Param param
    #int3* value
    return py_AcResult

def py_acDeviceLoadMeshInfo(const Device device, const AcMeshInfo device_config):
    #interface variables
    #const Device device
    #const AcMeshInfo device_config
    return py_AcResult

def py_acDeviceLoadDefaultUniforms(device):
    #interface variables
    #const Device device
    return py_AcResult

def py_acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh,
                                          vtxbuf_handle, src, dst,
                                          num_vertices):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcMesh host_mesh
    #const VertexBufferHandle vtxbuf_handle
    #const int3 src
    #const int3 dst
    #const int num_vertices
    return py_AcResult

def py_acDeviceLoadVertexBuffer(device, stream, host_mesh, vtxbuf_handle):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcMesh host_mesh
    #const VertexBufferHandle vtxbuf_handle
    return py_AcResult

def py_acDeviceLoadMesh(device, stream, host_mesh):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcMesh host_mesh
    return py_AcResult

def py_acDeviceSetVertexBuffer(device, stream, handle, value):
    #interface variables
    #const Device device
    #const Stream stream
    #const VertexBufferHandle handle
    #const AcReal value
    return py_AcResult

def py_acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src,
                                           dst, num_vertices, host_mesh):
    #interface variables
    #const Device device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #const int3 src
    #const int3 dst
    #const int num_vertices
    #AcMesh* host_mesh
    return py_AcResult

def py_acDeviceStoreVertexBuffer(device, stream, vtxbuf_handle, host_mesh):
    #interface variables
    #const Device device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #AcMesh* host_mesh
    return py_AcResult

def py_acDeviceStoreMesh(device, stream, host_mesh):
    #interface variables
    #const Device device
    #const Stream stream
    #AcMesh* host_mesh
    return py_AcResult

def py_acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle,
                                              src, dst, num_vertices,
                                              dst_device):
    #interface variables
    #const Device src_device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #const int3 src
    #const int3 dst
    #const int num_vertices
    #Device dst_device
    return py_AcResult

def py_acDeviceTransferMeshWithOffset(src_device, stream, src, dst,
                                      num_vertices, dst_device):
    #interface variables
    #const Device src_device
    #const Stream stream
    #const int3 src
    #const int3 dst
    #const int num_vertices
    #Device* dst_device
    return py_AcResult

def py_acDeviceTransferVertexBuffer(src_device, stream, vtxbuf_handle, dst_device):
    #interface variables
    #const Device src_device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #Device dst_device
    return py_AcResult

def py_acDeviceTransferMesh(src_device, stream, dst_device):
    #interface variables
    #const Device src_device
    #const Stream stream
    #Device dst_device
    return py_AcResult

def py_acDeviceIntegrateSubstep(device, stream, step_number, start, end, dt):
    #interface variables
    #const Device device
    #const Stream stream
    #const int step_number
    #const int3 start
    #const int3 end
    #const AcReal dt
    return py_AcResult

def py_acDevicePeriodicBoundcondStep(device, stream, vtxbuf_handle, start, end):
    #interface variables
    #const Device device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #const int3 start
    #const int3 end
    return py_AcResult

def py_acDevicePeriodicBoundconds(device, stream, start, end):
    #interface variables
    #const Device device
    #const Stream stream
    #const int3 start
    #const int3 end
    return py_AcResult

def py_acDeviceGeneralBoundcondStep(device, stream, vtxbuf_handle, start, end,
                                    config, bindex):
    #interface variables
    #const Device device
    #const Stream stream
    #const VertexBufferHandle vtxbuf_handle
    #const int3 start
    #const int3 end
    #const AcMeshInfo config
    #const int3 bindex
    return py_AcResult

def py_acDeviceGeneralBoundconds(device, stream, start, end, config, bindex):
    #interface variables
    #const Device device
    #const Stream stream
    #const int3 start
    #const int3 end
    #const AcMeshInfo config
    #const int3 bindex
    return py_AcResult

def py_acDeviceReduceScal(device, stream, rtype, vtxbuf_handle, result):
    #interface variables
    #const Device device
    #const Stream stream
    #const ReductionType rtype
    #const VertexBufferHandle vtxbuf_handle
    #AcReal* result
    return py_AcResult

def py_acDeviceReduceVec(device, stream_type, rtype, vtxbuf0, vtxbuf1,
                         vtxbuf2, result):
    #interface variables
    #const Device device
    #const Stream stream_type
    #const ReductionType rtype
    #const VertexBufferHandle vtxbuf0
    #const VertexBufferHandle vtxbuf1
    #const VertexBufferHandle vtxbuf2
    #AcReal* result
    return py_AcResult

def py_acDeviceReduceVecScal(device, stream_type, rtype, vtxbuf0, vtxbuf1,
                             vtxbuf2, vtxbuf3, result):
    #interface variables
    #const Device device
    #const Stream stream_type
    #const ReductionType rtype
    #const VertexBufferHandle vtxbuf0
    #const VertexBufferHandle vtxbuf1
    #const VertexBufferHandle vtxbuf2
    #const VertexBufferHandle vtxbuf3
    #AcReal* result
    return py_AcResult

def py_acDeviceRunMPITest():
    #interface variables
    #void
    return py_AcResult

def py_acDeviceLaunchKernel(device, stream, kernel, start, end):
    #interface variables
    #const Device device
    #const Stream stream
    #const Kernel kernel
    #const int3 start
    #const int3 end
    return py_AcResult

def py_acDeviceLoadStencil(device, stream, stencil, data):
    #interface variables
    #const Device device
    #const Stream stream
    #const Stencil stencil
    #const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]
    return py_AcResult

def py_acDeviceStoreStencil(device, stream, stencil, data):
    #interface variables
    #const Device device
    #const Stream stream
    #const Stencil stencil
    #AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]
    return py_AcResult

def py_acDeviceVolumeCopy(device, stream, in, in_offset, in_volume, out,
                          out_offset, out_volume):
    #interface variables
    #const Device device
    #const Stream stream
    #const AcReal* in
    #const int3 in_offset
    #const int3 in_volume
    #AcReal* out
    #const int3 out_offset
    #const int3 out_volume
    return py_AcResult

'''
=============================================================================
Helper functions
=============================================================================
'''
# Updates the built-in parameters based on nx, ny and nz
def py_acHostUpdateBuiltinParams(config):
    #interface variables
    #AcMeshInfo* config
    return py_AcResult

# Creates a mesh stored in host memory
def py_acHostMeshCreate(mesh_info, mesh):
    #interface variables
    #const AcMeshInfo mesh_info
    #AcMesh* mesh
    return py_AcResult

# Randomizes a host mesh
def py_acHostMeshRandomize(mesh):
    #interface variables
    #AcMesh* mesh
    return py_AcResult

# Destroys a mesh stored in host memory
def py_acHostMeshDestroy(mesh):
    #interface variables
    #AcMesh* mesh
    return py_AcResult
