#include "astaroth.h"
#include "kernels/kernels.h"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/find.h>


AcMeshCell
acDeviceMinElement(const Device device, const Stream stream, const Field field)
{
#if USE_HIP
  auto sync_exec_policy = thrust::cuda::par.on(device->streams[stream]);
#else
  auto sync_exec_policy = thrust::device;
#endif
 
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
#if USE_HIP
  auto sync_exec_policy = thrust::cuda::par.on(device->streams[stream]);
#else
  auto sync_exec_policy = thrust::device;
#endif

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

struct dev_is_a_nan
{
    __host__ __device__
    bool operator()(AcReal x)
    {
        return isnan(x);
    }
};

AcMeshBooleanSearchResult
acDeviceFirstNANElement(const Device device, const Stream stream, const Field field)
{
#if USE_HIP
  auto sync_exec_policy = thrust::cuda::par.on(device->streams[stream]);
#else
  auto sync_exec_policy = thrust::device;
#endif
 
  size_t field_length = acVertexBufferSize(device->local_config);
  thrust::device_ptr<AcReal> buffer_start = thrust::device_pointer_cast(device->vba.in[(size_t)field]);
  thrust::device_ptr<AcReal> buffer_end = thrust::device_pointer_cast(device->vba.in[(size_t)field]+field_length);


  thrust::device_ptr<AcReal> NAN_location = thrust::find_if(sync_exec_policy, buffer_start, buffer_end, dev_is_a_nan());
  int idx = NAN_location - buffer_start;

  size_t mx = device->local_config.int_params[AC_mx];
  size_t mxy = device->local_config.int_params[AC_mxy];
  int3 location{idx%mx, (idx%mxy)/mx, idx/mxy};

  //Still have to fetch the value
  AcMeshBooleanSearchResult result{location, NAN_location != buffer_end};

  return result;
}
