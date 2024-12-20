VertexBufferArray
acDeviceGetVBA(const Device device);

int
acDeviceGetId(const Device);

AcReduceBuffer
acDeviceGetProfileReduceBuffer(const Device, const Profile prof);

AcReal*
acDeviceGetProfileBuffer(const Device, const Profile prof);

acKernelInputParams*
acDeviceGetKernelInputParams(const Device device);

AcReal**
acDeviceGetStartOfProfiles(const Device device);

#include "device_set_output_decl.h"
#include "device_set_output_overloads.h"
