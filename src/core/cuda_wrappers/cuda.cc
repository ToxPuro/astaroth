#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "acreal.h"
#include "astaroth_cuda_wrappers.h"

#if AC_USE_HIP
#include <hip/hip_runtime_api.h>  // Streams

#include "hip.h"
#if PROFILING_ENABLED
#include <roctracer/roctracer_ext.h>  // Profiling
#endif
#else
#if PROFILING_ENABLED
#include <cuda_profiler_api.h>  // Profiling
#endif
#include <cuda_runtime_api.h>  // Streams
#endif

static inline void __attribute__((unused))
cuda_assert(cudaError_t code, const char* file, int line, bool should_abort)
{
  if (code != cudaSuccess) {
    time_t terr;
    time(&terr);
    fprintf(stderr, "%s", ctime(&terr));
    fprintf(stderr, "\tCUDA error in file %s line %d: %s\n", file, line,
            cudaGetErrorString(code));
    fflush(stderr);

    if (should_abort)
      abort();
  }
}

/*
 * =============================================================================
 * General error checking
 * =============================================================================
 */

#define ERRCHK_CUDA(params)                                                         \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }

#ifdef NDEBUG
#undef ERRCHK_CUDA
#define ERRCHK_CUDA(params) (void)(params)
#endif

cudaError_t
acDriverGetVersion(int* dst)
{
	const auto res = cudaDriverGetVersion(dst);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acRuntimeGetVersion(int* dst)
{
	const auto res = cudaRuntimeGetVersion(dst);
        ERRCHK_CUDA(res);

	return res;
}

cudaError_t
acStreamSynchronize(cudaStream_t stream)
{
	const auto res = cudaStreamSynchronize(stream);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acDeviceSynchronize()
{
	const auto res = cudaDeviceSynchronize();
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acSetDevice(const int id)
{
	const auto res = cudaSetDevice(id);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acGetDeviceCount(int* dst)
{
	const auto res = cudaGetDeviceCount(dst);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acDeviceSetSharedMemConfig(const cudaSharedMemConfig config)
{
	const auto res = cudaDeviceSetSharedMemConfig(config);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acStreamCreateWithPriority(cudaStream_t* dst, int option, int priority)
{
	const auto res = cudaStreamCreateWithPriority(dst,option,priority);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acStreamDestroy(cudaStream_t stream)
{
	const auto res = cudaStreamDestroy(stream);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemcpy(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind)
{
	const auto res = cudaMemcpy(dst,src,bytes,kind);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemcpy(void* dst, const void* src, const size_t bytes, cudaMemcpyKind kind)
{
	const auto res = cudaMemcpy(dst,src,bytes,kind);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemcpyAsync(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream)
{
	const auto res = cudaMemcpyAsync(dst,src,bytes,kind,stream);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemcpyAsync(void* dst, const void* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream)
{
	const auto res = cudaMemcpyAsync(dst,src,bytes,kind,stream);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemcpyPeerAsync(AcReal* dst, int dst_id, const AcReal* src, int src_id, const size_t bytes, const cudaStream_t stream)
{
	const auto res = cudaMemcpyPeerAsync(dst,dst_id,src,src_id,bytes,stream);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMemGetInfo(size_t* free_mem, size_t* total_mem)
{
	const auto res = cudaMemGetInfo(free_mem,total_mem);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acStreamQuery(cudaStream_t stream)
{
    const auto res = cudaStreamQuery(stream);
    return res;
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
	const auto res = cudaDeviceGetStreamPriorityRange(leastPriority,greatestPriority);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority)
{
	const auto res = cudaStreamCreateWithPriority(stream, flags, priority);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acMalloc(void** dst, const size_t bytes)
{
	const auto res = cudaMalloc(dst,bytes);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acFree(void* dst)
{
	const auto res = cudaFree(dst);
        ERRCHK_CUDA(res);
	return res;
}

cudaError_t
acFreeHost(void* dst)
{
	const auto res = cudaFreeHost(dst);
        ERRCHK_CUDA(res);
	return res;
}
cudaError_t
acMallocHost(void** dst, const size_t bytes)
{
	const auto res = cudaMallocHost(dst,bytes);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acGetDevice(int* dst)
{
	const auto res = cudaGetDevice(dst);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acGetLastError()
{
	const auto res = cudaGetLastError();
	return res;
}

cudaError_t
acEventCreate(cudaEvent_t* event)
{
	const auto res = cudaEventCreate(event);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acEventRecord(cudaEvent_t event)
{
	const auto res = cudaEventRecord(event);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acEventSynchronize(cudaEvent_t event)
{
	const auto res = cudaEventSynchronize(event);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acEventElapsedTime(float* time, cudaEvent_t start_event, cudaEvent_t end_event)
{
	const auto res = cudaEventElapsedTime(time,start_event,end_event);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acEventDestroy(cudaEvent_t event)
{
	const auto res = cudaEventDestroy(event);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t 
acGetDeviceProperties(cudaDeviceProp* prop, int device)
{
	const auto res = cudaGetDeviceProperties(prop,device);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t smemSize)
{
	const auto res = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks , func, blockSize, smemSize);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acDeviceGetAttribute(int* dst,  cudaDeviceAttr attr, int device)
{
	const auto res = cudaDeviceGetAttribute(dst,attr,device);
        ERRCHK_CUDA(res);

	return res;
}
cudaError_t
acLaunchCooperativeKernel(void* func,dim3 bpg,dim3 tpb,void** args,size_t smem,cudaStream_t stream);

cudaError_t
acLaunchCooperativeKernel(void* func,dim3 bpg,dim3 tpb,void** args,size_t smem,cudaStream_t stream)
{
	return cudaLaunchCooperativeKernel(func,bpg,tpb,args,smem,stream);
}
cudaError_t
acDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
	return cudaDeviceGetPCIBusId(pciBusId,len,device);
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
#if AC_USE_HIP
	cudaProfilerStart();
	return cudaSuccess;
#else
	return cudaProfilerStart();
#endif
}
cudaError_t
acProfilerStop()
{
#if AC_USE_HIP
	cudaProfilerStop();
	return cudaSuccess;
#else
	return cudaProfilerStop();
#endif
}
#endif
