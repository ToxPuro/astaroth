#MV: Basic principle draft based on the Fortran interface. Does not work yet. 

import astaroth

# AcMeshInfo object / datatype requires a combatible form and translation step.
info = astaroth.AcMeshInfo()

# We can let acdevicecreate set device pointer. No need for separate unit definition. 

# How to get C #defines with into python.
#print("Num int params: %i", AC_NUM_INT_PARAMS)
#Possible principle:
print("Num int params: %i", astaroth.cdefines("AC_NUM_INT_PARAMS"))

info.set_nxnynz_dims = (128, 128, 128)

# device needs to be a functional C pointer
device = astaroth.acDeviceCreate(0, info)
astaroth.acDevicePrintInfo(device)
astaroth.acDeviceDestroy(device)

