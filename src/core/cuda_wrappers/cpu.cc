#include "device_headers.h"
#include "acreal.h"
#include <cstring>
#include <stdio.h>
#include <limits>
#include <stdlib.h>
#include "host_datatypes.h"

cudaError_t
acStreamSynchronize(cudaStream_t){return cudaSuccess;}
cudaError_t
acDeviceSynchronize(){return cudaSuccess;}
cudaError_t
acSetDevice(const int){return cudaSuccess;}
cudaError_t
acGetDeviceCount(int* dst){*dst = 1; return cudaSuccess;}
cudaError_t
acDeviceSetSharedMemConfig(const cudaSharedMemConfig){return cudaSuccess;}
cudaError_t
acStreamCreateWithPriority(cudaStream_t*, int, int){return cudaSuccess;}
cudaError_t
acStreamDestroy(cudaStream_t){return cudaSuccess;}
cudaError_t
acMemcpy(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind)
{
	memcpy(dst,src,bytes);
	return cudaSuccess;
}
cudaError_t
acMemcpy(void* dst, const void* src, const size_t bytes, cudaMemcpyKind)
{
	memcpy(dst,src,bytes);
	return cudaSuccess;
}
cudaError_t
acMemcpyAsync(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind, const cudaStream_t)
{
	memcpy(dst,src,bytes);
	return cudaSuccess;
}

cudaError_t
acMemcpyAsync(void* dst, const void* src, const size_t bytes, cudaMemcpyKind, const cudaStream_t)
{
	memcpy(dst,src,bytes);
	return cudaSuccess;
}

cudaError_t
acMemcpyPeerAsync(AcReal* dst, int, const AcReal* src, int, const size_t bytes, const cudaStream_t)
{
	memcpy(dst,src,bytes);
	return cudaSuccess;
}
cudaError_t
acMemGetInfo(size_t* free_mem, size_t* total_mem)
{
	*free_mem  = std::numeric_limits<size_t>::max();
	*total_mem = std::numeric_limits<size_t>::max();
	return cudaSuccess;
}
cudaError_t
acStreamQuery(cudaStream_t)
{
	return cudaSuccess;
}
const char*
acGetErrorString(cudaError_t)
{
	return "CPU-BUILD!";
}
const char*
acGetErrorName(cudaError_t)
{
	return "CPU-BUILD!";
}
cudaError_t
acDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
	*leastPriority    = 0;
	*greatestPriority = 0;
	return cudaSuccess;
}
cudaError_t
acStreamCreateWithPriority(cudaStream_t*, unsigned int, int){return cudaSuccess;}
cudaError_t
acMalloc(void** dst, const size_t bytes)
{
	*dst = malloc(bytes);
	return cudaSuccess;
}
cudaError_t
acFree(void* dst)
{
	free(dst);
	return cudaSuccess;
}
cudaError_t
acMallocHost(void** dst, const size_t bytes)
{
	*dst = malloc(bytes);
	return cudaSuccess;
}
cudaError_t
acGetDevice(int* dst)
{
	*dst = 0;
	return cudaSuccess;
}
cudaError_t
acGetLastError()
{
	return cudaSuccess;
}
cudaError_t
acEventCreate(cudaEvent_t*){return cudaSuccess;}
cudaError_t
acEventRecord(cudaEvent_t){return cudaSuccess;}
cudaError_t
acEventSynchronize(cudaEvent_t){return cudaSuccess;}
cudaError_t
acEventElapsedTime(float* time, cudaEvent_t, cudaEvent_t) 
{
	*time = 0.0;
	return cudaSuccess;
}
cudaError_t
acEventDestroy(cudaEvent_t){return cudaSuccess;}
cudaError_t 
acGetDeviceProperties(cudaDeviceProp* prop, int)
{
	(void)prop;
	return cudaSuccess;
}
cudaError_t
acOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void*, int, size_t)
{
	*numBlocks = 1;
	return cudaSuccess;
}
cudaError_t
acDeviceGetAttribute(int* dst, cudaDeviceAttr, int)
{
	*dst = 1;
	return cudaSuccess;
}
cudaError_t
acLaunchCooperativeKernel(void*,dim3,dim3,void**,size_t,cudaStream_t)
{
	return cudaSuccess;
}
cudaError_t
acMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind)
{
	memcpy((void*)((char*)symbol+offset),src,count);
	return cudaSuccess;
}
cudaError_t
acMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind, cudaStream_t)
{
	memcpy((void*)((char*)symbol+offset),src,count);
	return cudaSuccess;
}
cudaError_t
acDeviceGetPCIBusId(char* pciBusId, int, int)
{
	sprintf(pciBusId,"CPU-BUILD");
	return cudaSuccess;
}
cudaError_t 
acMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind)
{
	memcpy(dst,(void*)((char*)symbol+offset),count);
	return cudaSuccess;
}
cudaError_t 
acMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind, cudaStream_t)
{
	memcpy(dst,(void*)((char*)symbol+offset),count);
	return cudaSuccess;
}

cudaError_t 
acPeekAtLastError()
{
	return cudaSuccess;
}
