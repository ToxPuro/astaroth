#include "astaroth.h"
#include "kernels/kernels.h"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>


AcMeshCell
acDeviceMinElement(const Device device, const Stream stream, const Field field)
{
  auto sync_exec_policy = thrust::cuda::par.on(device->streams[stream]);
 
  size_t field_length = acVertexBufferSize(device->local_config);
  thrust::device_ptr<AcReal> buffer_start = thrust::device_pointer_cast(device->vba.in[(size_t)field]);
  thrust::device_ptr<AcReal> buffer_end = thrust::device_pointer_cast(device->vba.in[(size_t)field]+field_length);

  thrust::device_ptr<AcReal> min_elem = thrust::min_element(sync_exec_policy, buffer_start, buffer_end);
  //thrust::device_ptr<AcReal> min_elem = thrust::min_element(buffer_start, buffer_end);
  int idx = min_elem - buffer_start;

  size_t mx = device->local_config.int_params[AC_mx];
  size_t mxy = device->local_config.int_params[AC_mxy];
  int3 location{idx%mx, (idx%mxy)/mx, idx/mxy};

  //Still have to fetch the value
  AcMeshCell result{location, *min_elem};

  return result;
}

AcMeshCell
acDeviceMaxElement(const Device device, const Stream stream, const Field field)
{
  auto sync_exec_policy = thrust::cuda::par.on(device->streams[stream]);
 
  size_t field_length = acVertexBufferSize(device->local_config);
  thrust::device_ptr<AcReal> buffer_start = thrust::device_pointer_cast(device->vba.in[(size_t)field]);
  thrust::device_ptr<AcReal> buffer_end = thrust::device_pointer_cast(device->vba.in[(size_t)field]+field_length);

  thrust::device_ptr<AcReal> max_elem = thrust::max_element(sync_exec_policy, buffer_start, buffer_end);
  int idx = max_elem - buffer_start;

  size_t mx = device->local_config.int_params[AC_mx];
  size_t mxy = device->local_config.int_params[AC_mxy];
  int3 location{idx%mx, (idx%mxy)/mx, idx/mxy};

  //Still have to fetch the value
  AcMeshCell result{location, *max_elem};

  return result;
}

