#include "acc_runtime.h"
#include "kernels.h"
#include "math_utils.h"
#include "astaroth_cuda_wrappers.h"
#include "ac_buffer.h"

template<typename T1, typename T2, typename T3>
AcResult
acLaunchKernelVariadic1d(AcKernel kernel, const cudaStream_t stream, const size_t start, const size_t end,T1 param1, T2 param2, T3 param3)
{
  const Volume volume_start = {start,0,0};
  const Volume volume_end   = {end,1,1};
  VertexBufferArray vba{};
  acLoadKernelParams(vba.on_device.kernel_input_params,kernel,param1,param2,param3); 
  return acLaunchKernelWithTPB(kernel,stream,volume_start,volume_end,vba,dim3(256,1,1));
}

template<typename T1, typename T2>
AcResult
acLaunchKernelVariadic1d(AcKernel kernel, const cudaStream_t stream, const size_t start, const size_t end,T1 param1, T2 param2)
{
  const Volume volume_start = {start,0,0};
  const Volume volume_end   = {end,1,1};
  VertexBufferArray vba{};
  acLoadKernelParams(vba.on_device.kernel_input_params,kernel,param1,param2); 
  return acLaunchKernelWithTPB(kernel,stream,volume_start,volume_end,vba,dim3(256,1,1));
}

AcResult
acKernelFlushReal(const cudaStream_t stream, AcReal* arr, const size_t n,
              const AcReal value)
{
  ERRCHK_ALWAYS(arr || n == 0);
  if(n == 0) return AC_SUCCESS;
  acLaunchKernelVariadic1d(AC_FLUSH_REAL,stream,0,n,arr,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

AcResult
acKernelFlushComplex(const cudaStream_t stream, AcComplex* arr, const size_t n,
              const AcComplex value)
{
  ERRCHK_ALWAYS(arr || n == 0);
  if(n == 0) return AC_SUCCESS;
  acLaunchKernelVariadic1d(AC_FLUSH_COMPLEX,stream,0,n,arr,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

AcResult
acKernelFlushInt(const cudaStream_t stream, int* arr, const size_t n,
              const int value)
{
  ERRCHK_ALWAYS(arr || n == 0);
  if(n == 0) return AC_SUCCESS;
  acLaunchKernelVariadic1d(AC_FLUSH_INT,stream,0,n,arr,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

AcResult
acKernelFlushFloat(const cudaStream_t stream, float* arr, const size_t n,
              const float value)
{
  ERRCHK_ALWAYS(arr || n == 0);
  if(n == 0) return AC_SUCCESS;
  acLaunchKernelVariadic1d(AC_FLUSH_FLOAT,stream,0,n,arr,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

AcResult
acKernelFlush(const cudaStream_t stream, AcReal* arr, const size_t n,
              const AcReal value)
{
	return acKernelFlushReal(stream,arr,n,value);
}

AcResult
acKernelFlush(const cudaStream_t stream, int* arr, const size_t n,
              const int value)
{
	return acKernelFlushInt(stream,arr,n,value);
}
AcResult
acKernelFlush(const cudaStream_t stream, AcComplex* arr, const size_t n,
              const AcComplex value)
{
	return acKernelFlushComplex(stream,arr,n,value);
}

#if AC_DOUBLE_PRECISION
AcResult
acKernelFlush(const cudaStream_t stream, float* arr, const size_t n,
              const float value)
{
	return acKernelFlushFloat(stream,arr,n,value);
}
#endif

AcResult
acMultiplyInplaceComplex(const AcReal value, const size_t count, AcComplex* array)
{
  acLaunchKernelVariadic1d(AC_MULTIPLY_INPLACE_COMPLEX,0,size_t(0),count,value,array);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acMultiplyInplace(const AcReal value, const size_t count, AcReal* array)
{
  acLaunchKernelVariadic1d(AC_MULTIPLY_INPLACE,0,size_t(0),count,value,array);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acKernelVolumeCopy(const cudaStream_t stream,                                    //
                   const AcReal* in, const Volume in_offset, const Volume in_volume, //
                   AcReal* out, const Volume out_offset, const Volume out_volume)
{
    VertexBufferArray vba{};
    acLoadKernelParams(vba.on_device.kernel_input_params,AC_VOLUME_COPY,(AcReal*)in,in_offset,in_volume,out,out_offset,out_volume); 
    const Volume start = {0,0,0};
    const Volume nn = to_volume(min(to_int3(in_volume), to_int3(out_volume)));
    acLaunchKernel(AC_VOLUME_COPY,stream,start,nn,vba);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acComplexToReal(const AcComplex* src, const size_t count, AcReal* dst)
{
  acLaunchKernelVariadic1d(AC_COMPLEX_TO_REAL,0,size_t(0),count,(AcComplex*)src,dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acRealToComplex(const AcReal* src, const size_t count, AcComplex* dst)
{
  acLaunchKernelVariadic1d(AC_REAL_TO_COMPLEX,0,size_t(0),count,(AcReal*)src,dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acPlanarToComplex(const AcReal* real_src, const AcReal* imag_src,const size_t count, AcComplex* dst)
{
  acLaunchKernelVariadic1d(AC_PLANAR_TO_COMPLEX,0,size_t(0),count,(AcReal*)real_src,(AcReal*)imag_src,dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acComplexToPlanar(const AcComplex* src,const size_t count,AcReal* real_dst,AcReal* imag_dst)
{
  acLaunchKernelVariadic1d(AC_COMPLEX_TO_PLANAR,0,size_t(0),count,(AcComplex*)src,real_dst,imag_dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}
