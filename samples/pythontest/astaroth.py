from ctypes import *
astaroth = cdll.LoadLibrary('./libastaroth_core_shared.so')

NUM_INT_PARAMS = 20
NUM_INT3_PARAMS = 2
NUM_REAL_PARAMS = 74
NUM_REAL3_PARAMS = 0
NUM_VTXBUF_HANDLES = 8

AC_DOUBLE_PRECISION = True

c_real = c_float
if AC_DOUBLE_PRECISION:
    c_real = c_double

class AcMeshInfo(Structure):
    _fields_ = [("int_params", NUM_INT_PARAMS * c_int),
                ("int_params", 3 * NUM_INT3_PARAMS * c_int),
                ("real_params", NUM_REAL_PARAMS * c_real),
                ("real3_params", 3 * NUM_REAL3_PARAMS * c_real)]

info = AcMeshInfo()
device = astaroth.acDeviceCreate(0, info);
astaroth.acDevicePrintInfo(device)
astarth.acDeviceDestroy(device)
