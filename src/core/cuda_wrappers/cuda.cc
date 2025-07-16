#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime_api.h> // Streams
#if PROFILING_ENABLED
#include <roctracer/roctracer_ext.h>       // Profiling
#endif

#else 
#if PROFILING_ENABLED
#include <cuda_profiler_api.h> // Profiling
#endif
#include <cuda_runtime_api.h>  // Streams
#endif

#include "acreal.h"
#include "astaroth_cuda_wrappers.h"

cudaError_t
acStreamSynchronize(cudaStream_t stream)
{
	return cudaStreamSynchronize(stream);
}
cudaError_t
acDeviceSynchronize()
{
	return cudaDeviceSynchronize();
}
cudaError_t
acSetDevice(const int id)
{
	return cudaSetDevice(id);
}
cudaError_t
acGetDeviceCount(int* dst)
{
	return cudaGetDeviceCount(dst);
}
cudaError_t
acDeviceSetSharedMemConfig(const cudaSharedMemConfig config)
{
	return cudaDeviceSetSharedMemConfig(config);
}
cudaError_t
acStreamCreateWithPriority(cudaStream_t* dst, int option, int priority)
{
	return cudaStreamCreateWithPriority(dst,option,priority);
}
cudaError_t
acStreamDestroy(cudaStream_t stream)
{
	return cudaStreamDestroy(stream);
}
cudaError_t
acMemcpy(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind)
{
	return cudaMemcpy(dst,src,bytes,kind);
}
cudaError_t
acMemcpy(void* dst, const void* src, const size_t bytes, cudaMemcpyKind kind)
{
	return cudaMemcpy(dst,src,bytes,kind);
}
cudaError_t
acMemcpyAsync(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream)
{
	return cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
cudaError_t
acMemcpyAsync(void* dst, const void* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream)
{
	return cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
cudaError_t
acMemcpyPeerAsync(AcReal* dst, int dst_id, const AcReal* src, int src_id, const size_t bytes, const cudaStream_t stream)
{
	return cudaMemcpyPeerAsync(dst,dst_id,src,src_id,bytes,stream);
}
cudaError_t
acMemGetInfo(size_t* free_mem, size_t* total_mem)
{
	return cudaMemGetInfo(free_mem,total_mem);
}
cudaError_t
acStreamQuery(cudaStream_t stream)
{
    return cudaStreamQuery(stream);
}
const char*
acGetErrorString(cudaError_t err)
{
    return cudaGetErrorString(err);
}

const char*
acGetErrorName(cudaError_t err)
{
    return cudaGetErrorName(err);
}
cudaError_t
acDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
	return cudaDeviceGetStreamPriorityRange(leastPriority,greatestPriority);
}
cudaError_t
acStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority)
{
	return cudaStreamCreateWithPriority(stream, flags, priority);
}
cudaError_t
acMalloc(void** dst, const size_t bytes)
{
	return cudaMalloc(dst,bytes);
}
cudaError_t
acFree(void* dst)
{
	return cudaFree(dst);
}
cudaError_t
acMallocHost(void** dst, const size_t bytes)
{
	return cudaMallocHost(dst,bytes);
}
cudaError_t
acGetDevice(int* dst)
{
	return cudaGetDevice(dst);
}
cudaError_t
acGetLastError()
{
	return cudaGetLastError();
}

cudaError_t
acEventCreate(cudaEvent_t* event)
{
	return cudaEventCreate(event);
}
cudaError_t
acEventRecord(cudaEvent_t event)
{
	return cudaEventRecord(event);
}
cudaError_t
acEventSynchronize(cudaEvent_t event)
{
	return cudaEventSynchronize(event);
}
cudaError_t
acEventElapsedTime(float* time, cudaEvent_t start_event, cudaEvent_t end_event)
{
	return cudaEventElapsedTime(time,start_event,end_event);
}
cudaError_t
acEventDestroy(cudaEvent_t event)
{
	return cudaEventDestroy(event);
}
cudaError_t 
acGetDeviceProperties(cudaDeviceProp* prop, int device)
{
	return cudaGetDeviceProperties(prop,device);
}
cudaError_t
acOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t smemSize)
{
	return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks , func, blockSize, smemSize);
}
cudaError_t
acDeviceGetAttribute(int* dst,  cudaDeviceAttr attr, int device)
{
	return cudaDeviceGetAttribute(dst,attr,device);
}
cudaError_t
acLaunchCooperativeKernel(void* func,dim3 bpg,dim3 tpb,void** args,size_t smem,cudaStream_t stream);

cudaError_t
acLaunchCooperativeKernel(void* func,dim3 bpg,dim3 tpb,void** args,size_t smem,cudaStream_t stream)
{
	return cudaLaunchCooperativeKernel(func,bpg,tpb,args,smem,stream);
}
cudaError_t
acMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind)
{
	return cudaMemcpyToSymbol(symbol,src,count,offset,kind);
}
cudaError_t
acMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)
{
	return cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
}
cudaError_t
acDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
	return cudaDeviceGetPCIBusId(pciBusId,len,device);
}
cudaError_t 
acMemcpyFromSymbol( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind)
{
	return cudaMemcpyFromSymbol(dst,symbol,count,offset,kind);
}
cudaError_t 
acMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)
{
	return cudaMemcpyFromSymbolAsync(dst,symbol,count,offset,kind,stream);
}
cudaError_t 
acPeekAtLastError()
{
	return cudaPeekAtLastError();
}
#if PROFILING_ENABLED
cudaError_t
acProfilerStart()
{
	return cudaProfilerStart();
}
cudaError_t
acProfilerStop()
{
	return cudaProfilerStop();
}
#endif
