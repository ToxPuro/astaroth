VertexBufferArray
acDeviceGetVBA(const Device device);

int
acDeviceGetId(const Device);

AcReal*
acDeviceGetProfileReduceScratchpad(const Device, const Profile prof);

AcReal*
acDeviceGetProfileBuffer(const Device, const Profile prof);

AcReal**
acDeviceGetProfileCubTmp(const Device device,Profile prof);

size_t*
acDeviceGetProfileCubTmpSize(const Device device,const Profile prof);

acKernelInputParams*
acDeviceGetKernelInputParams(const Device device);

AcReal**
acDeviceGetStartOfProfiles(const Device device);

#include "device_set_output_decl.h"
#include "device_set_output_overloads.h"
