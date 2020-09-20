from ctypes import *
astaroth = cdll.LoadLibrary('../../build/src/core/libastaroth_core_shared.so')

NUM_INT_PARAMS = 20
NUM_INT3_PARAMS = 2
NUM_REAL_PARAMS = 74
NUM_REAL3_PARAMS = 0
NUM_VTXBUF_HANDLES = 8

AC_DOUBLE_PRECISION = True

AC_nx = 0
AC_ny = 1
AC_nz = 2

c_real = c_float
if AC_DOUBLE_PRECISION:
    c_real = c_double

class AcMeshInfo(Structure):
    _fields_ = [("int_params", NUM_INT_PARAMS * c_int),
                ("int3_params", 3 * NUM_INT3_PARAMS * c_int),
                ("real_params", NUM_REAL_PARAMS * c_real),
                ("real3_params", 3 * NUM_REAL3_PARAMS * c_real)]

info = AcMeshInfo()
info.int_params[AC_nx] = 128
info.int_params[AC_ny] = 128
info.int_params[AC_nz] = 128
astaroth.acUpdateBuiltinParams(pointer(info))
astaroth.acPrintMeshInfo(info)

device = c_void_p(1)
astaroth.acDeviceCreate(0, info, byref(device));
astaroth.acDevicePrintInfo(device)
astaroth.acDeviceDestroy(device)
