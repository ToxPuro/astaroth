
import ctypes 
import pathlib

libname = pathlib.Path().absolute() / "libastaroth.so"
c_lib = ctypes.CDLL(libname)

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

mesh_info = py_load_config(config_path)

mesh      = py_acmesh_create(mesh_info)

mesh      = py_initialize_mesh("random_fields", mesh)

device = py_acDeviceCreate(device_number, mesh_info)
py_acDevicePrintInfo(device)
mesh = py_acDeviceLoadMesh(device, mesh)

py_acDeviceDestroy(device)

