#include "astaroth.h"
#include "ac_helpers.h"
#include <sys/resource.h>
#include "astaroth_cuda_wrappers.h"

#if USE_COMPRESSIBLE_MEMORY
#include <cuda.h>
static cudaError_t
mallocCompressible(void** addr, const size_t requested_bytes)
{
  CUdevice device;
  ERRCHK_ALWAYS(cuCtxGetDevice(&device) == CUDA_SUCCESS);

  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id                = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

  size_t granularity;
  ERRCHK_CU_ALWAYS(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // Pad to align
  const size_t bytes = ((requested_bytes - 1) / granularity + 1) * granularity;

  CUdeviceptr dptr;
  ERRCHK_ALWAYS(cuMemAddressReserve(&dptr, bytes, 0, 0, 0) == CUDA_SUCCESS);

  CUmemGenericAllocationHandle handle;
  ERRCHK_ALWAYS(cuMemCreate(&handle, bytes, &prop, 0) == CUDA_SUCCESS)

  // Check if cuMemCreate was able to allocate compressible memory.
  CUmemAllocationProp alloc_prop;
  memset(&alloc_prop, 0, sizeof(CUmemAllocationProp));
  cuMemGetAllocationPropertiesFromHandle(&alloc_prop, handle);
  ERRCHK_ALWAYS(alloc_prop.allocFlags.compressionType ==
                CU_MEM_ALLOCATION_COMP_GENERIC);

  ERRCHK_ALWAYS(cuMemMap(dptr, bytes, 0, handle, 0) == CUDA_SUCCESS);
  ERRCHK_ALWAYS(cuMemRelease(handle) == CUDA_SUCCESS);

  CUmemAccessDesc accessDescriptor;
  accessDescriptor.location.id   = prop.location.id;
  accessDescriptor.location.type = prop.location.type;
  accessDescriptor.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  ERRCHK_ALWAYS(cuMemSetAccess(dptr, bytes, &accessDescriptor, 1) ==
                CUDA_SUCCESS);

  *addr = (void*)dptr;
  return cudaSuccess;
}

static void
freeCompressible(void* ptr, const size_t requested_bytes)
{
  CUdevice device;
  ERRCHK_ALWAYS(cuCtxGetDevice(&device) == CUDA_SUCCESS);

  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id                = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

  size_t granularity = 0;
  ERRCHK_ALWAYS(cuMemGetAllocationGranularity(
                    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) ==
                CUDA_SUCCESS);
  const size_t bytes = ((requested_bytes - 1) / granularity + 1) * granularity;

  ERRCHK_ALWAYS(ptr);
  ERRCHK_ALWAYS(cuMemUnmap((CUdeviceptr)ptr, bytes) == CUDA_SUCCESS);
  ERRCHK_ALWAYS(cuMemAddressFree((CUdeviceptr)ptr, bytes) == CUDA_SUCCESS);
}
#define ERRCHK_CU_ALWAYS(x) ERRCHK_ALWAYS((x) == CUDA_SUCCESS)
#endif

void
acDeviceMalloc(void** dst, const size_t bytes)
{
  if(acGetAmountOfDeviceMemoryFree() < bytes)
  {
	fprintf(stderr,"Tried to allocate %ld bytes but have only %ld bytes of memory left on the device\n", bytes, acGetAmountOfDeviceMemoryFree());
  	ERRCHK_ALWAYS(acGetAmountOfDeviceMemoryFree() >= bytes);
  }
 #if USE_COMPRESSIBLE_MEMORY 
    ERRCHK_CUDA_ALWAYS(mallocCompressible(dst, bytes));
 #else
    ERRCHK_CUDA_ALWAYS(acMalloc(dst, bytes));
  #endif
  ERRCHK_ALWAYS(dst != NULL);
}

void
acDeviceFree(void** dst, const int bytes)
{
#if USE_COMPRESSIBLE_MEMORY
  freeCompressible(*dst, bytes);
#else
  ERRCHK_CUDA_ALWAYS(acFree(*dst));
  //used to silence unused warning
  (void)bytes;
#endif
  *dst = NULL;
}
void
acDeviceFree(AcReal** dst, const int bytes)
{
	acDeviceFree((void**)dst,bytes);
}
void
acDeviceFree(AcComplex** dst, const int bytes)
{
	acDeviceFree((void**)dst,bytes);
}



