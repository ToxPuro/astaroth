/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#define AC_INSIDE_AC_LIBRARY 

#include "acc_runtime.h"
#include "../acc/string_vec.h"
typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);
#define AcReal3(x,y,z)   (AcReal3){x,y,z}
#define AcComplex(x,y)   (AcComplex){x,y}

static AcBool3 dimension_inactive{};
#include <math.h> 
#include <vector> // tbconfig

#include "errchk.h"
#include "math_utils.h"
#include <unordered_map>
#include <utility>
#include <sys/stat.h>

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <rocprim/rocprim.hpp>
#endif

#include "user_kernel_declarations.h"
#include "kernel_reduce_info.h"


#define USE_COMPRESSIBLE_MEMORY (0)

//TP: unfortunately cannot use color output since it might not be supported in each env
const bool useColor = false;

#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define RESET "\033[0m"

#define COLORIZE(symbol, color) (useColor ? color symbol RESET : symbol)


#include "acc/implementation.h"

static dim3 last_tpb = (dim3){0, 0, 0};
struct Int3Hash {
    std::size_t operator()(const int3& v) const {
        return std::hash<int>()(v.x) ^ std::hash<int>()(v.y) << 1 ^ std::hash<int>()(v.z) << 2;
    }
};
std::array<std::unordered_map<int3,int,Int3Hash>,NUM_KERNELS> reduce_offsets;
int kernel_running_reduce_offsets[NUM_KERNELS];

static int grid_pid = 0;
[[maybe_unused]] static int nprocs   = 0;
static AcMeasurementGatherFunc gather_func  = NULL;

#if AC_MPI_ENABLED
AcResult
acInitializeRuntimeMPI(const int _grid_pid,const int _nprocs, const AcMeasurementGatherFunc mpi_gather_func)
{
	grid_pid = _grid_pid;
	nprocs   = _nprocs;
	gather_func = mpi_gather_func;
	return AC_SUCCESS;
}
#endif

Volume
acKernelLaunchGetLastTPB(void)
{
  return to_volume(last_tpb);
}
int
acGetKernelReduceScratchPadSize(const AcKernel kernel)
{
	return kernel_running_reduce_offsets[(int)kernel];
}
int
acGetKernelReduceScratchPadMinSize()
{
	int res = 0; 
	for(int i = 0; i < NUM_KERNELS; ++i)
		res = (res < kernel_running_reduce_offsets[i]) ? kernel_running_reduce_offsets[i] : res;
	return res;
}
Volume
get_bpg(Volume dims, const Volume tpb)
{
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING:             // Fallthrough
  case EXPLICIT_CACHING:             // Fallthrough
  case EXPLICIT_CACHING_3D_BLOCKING: // Fallthrough
  case EXPLICIT_CACHING_4D_BLOCKING: // Fallthrough
  case EXPLICIT_PINGPONG_txw:        // Fallthrough
  case EXPLICIT_PINGPONG_txy:        // Fallthrough
  case EXPLICIT_PINGPONG_txyblocked: // Fallthrough
  case EXPLICIT_PINGPONG_txyz:       // Fallthrough
  case EXPLICIT_ROLLING_PINGPONG: {
    return (Volume){
        (size_t)ceil(1. * dims.x / tpb.x),
        (size_t)ceil(1. * dims.y / tpb.y),
        (size_t)ceil(1. * dims.z / tpb.z),
    };
  }
  default: {
    ERROR("Invalid IMPLEMENTATION in get_bpg");
    return (Volume){0, 0, 0};
  }
  }
}

size_t
get_warp_size()
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  return props.warpSize;
}


bool
is_valid_configuration(const Volume dims, const Volume tpb, const AcKernel)
{
  const size_t warp_size = get_warp_size();
  const size_t xmax      = (size_t)(warp_size * ceil(1. * dims.x / warp_size));
  const size_t ymax      = (size_t)(warp_size * ceil(1. * dims.y / warp_size));
  const size_t zmax      = (size_t)(warp_size * ceil(1. * dims.z / warp_size));
  const bool too_large   = (tpb.x > xmax) || (tpb.y > ymax) || (tpb.z > zmax);

  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING: {

	    if (too_large)
	      return false;

	    return true;
	  }
	  case EXPLICIT_CACHING_4D_BLOCKING: // Fallthrough
	    if (tpb.z > 1) return false;
	    [[fallthrough]];
	  case EXPLICIT_CACHING: // Fallthrough
	  case EXPLICIT_CACHING_3D_BLOCKING: {

	    // For some reason does not work without this
	    // Probably because of break vs continue when fetching (some threads
	    // quit too early if the dims are not divisible)
	    return !(dims.x % tpb.x) && !(dims.y % tpb.y) && !(dims.z % tpb.z);
	  }
	  case EXPLICIT_PINGPONG_txw: {
	    return (tpb.y == 1) && (tpb.z == 1);
	  }
	  case EXPLICIT_PINGPONG_txy: {
	    return (tpb.z == 1);
	  }
	  case EXPLICIT_PINGPONG_txyblocked: {
	    return (tpb.z == 1);
	  }
	  case EXPLICIT_PINGPONG_txyz: {
	    return true;
	  }
	  case EXPLICIT_ROLLING_PINGPONG: {
	    // OK for every other rolling pingpong implementation
	    // return true;

	    // Required only when unrolling smem loads
	    // Ensures two unrolls is enough to fill the smem buffer
	    return (2 * tpb.x >= STENCIL_WIDTH - 1 + tpb.x) &&
		   (2 * tpb.y >= STENCIL_HEIGHT - 1 + tpb.y);
	  }
	  default: {
	    ERROR("Invalid IMPLEMENTATION in is_valid_configuration");
    return false;
  }
  }
}

size_t
get_smem(const Volume tpb, const size_t stencil_order,
         const size_t bytes_per_elem)
{
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING: {
    return 0;
  }
  case EXPLICIT_CACHING: {
    return (tpb.x + stencil_order) * (tpb.y + stencil_order) * tpb.z *
           bytes_per_elem;
  }
  case EXPLICIT_CACHING_3D_BLOCKING: {
    return (tpb.x + stencil_order) * (tpb.y + stencil_order) *
           (tpb.z + stencil_order) * bytes_per_elem;
  }
  case EXPLICIT_CACHING_4D_BLOCKING: {
    return (tpb.x + stencil_order) * (tpb.y + stencil_order) * tpb.z *
           (NUM_FIELDS)*bytes_per_elem;
  }
  case EXPLICIT_PINGPONG_txw: {
    return 2 * (tpb.x + stencil_order) * NUM_FIELDS * bytes_per_elem;
  }
  case EXPLICIT_PINGPONG_txy: {
    return 2 * (tpb.x + stencil_order) * (tpb.y + stencil_order) *
           bytes_per_elem;
  }
  case EXPLICIT_PINGPONG_txyblocked: {
    const size_t block_size = 7;
    return 2 * (tpb.x + stencil_order) * (tpb.y + stencil_order) * block_size *
           bytes_per_elem;
  }
  case EXPLICIT_PINGPONG_txyz: {
    return 2 * (tpb.x + stencil_order) * (tpb.y + stencil_order) *
           (tpb.z + stencil_order) * bytes_per_elem;
  }
  case EXPLICIT_ROLLING_PINGPONG: {
    // tpbxy slices with halos
    // tpbz depth + 1 rolling cache slab
    return EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE * (tpb.x + stencil_order) *
           (tpb.y + stencil_order) * (tpb.z + 1) * bytes_per_elem;
  }
  default: {
    ERROR("Invalid IMPLEMENTATION in get_smem");
    return (size_t)-1;
  }
  }
}

/*
// Device info (TODO GENERIC)
// Use the maximum available reg count per thread
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#if AC_DOUBLE_PRECISION
#define MAX_THREADS_PER_BLOCK                                                  \
  (MAX_REGISTERS_PER_BLOCK / REGISTERS_PER_THREAD / 2)
#else
#define MAX_THREADS_PER_BLOCK (MAX_REGISTERS_PER_BLOCK / REGISTERS_PER_THREAD)
#endif
*/

__device__ __constant__ AcMeshInfoScalars d_mesh_info;
#include "dconst_arrays_decl.h"
//TP: We do this ugly macro because I want to keep the generated headers the same if we are compiling cpu analysis and for the actual gpu comp
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME) __device__ __constant__ DATATYPE* AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME 
#define DECLARE_CONST_DIMS_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, LEN) __device__ DATATYPE AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME[LEN]
#include "gmem_arrays_decl.h"



//The macros above generate d arrays like these:

// Astaroth 2.0 backwards compatibility START
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])


#define DEVICE_INLINE __device__ __forceinline__
#include "dconst_decl.h"



#include "get_address.h"
#include "load_dconst_arrays.h"
#include "store_dconst_arrays.h"


#define DEVICE_VTXBUF_IDX(i, j, k)                                             \
  ((i) + (j)*VAL(AC_mlocal).x + (k)*VAL(AC_mlocal_products).xy)

__device__ int
LOCAL_COMPDOMAIN_IDX(const int3 coord)
{
  return (coord.x) + (coord.y) * VAL(AC_nlocal).x + (coord.z) * VAL(AC_nlocal_products).xy;
}

#define print printf                          // TODO is this a good idea?
// passes an array into a device function and then calls len (need to modify
// the compiler to always pass arrays to functions as references before
// re-enabling)

#include "random.cuh"

#define suppress_unused_warning(X) (void)X
#define longlong long long
#define size(arr) (int)(sizeof(arr) / sizeof(arr[0])) // Leads to bugs if the user
#include "user_kernels.h"
#undef size
#undef longlong


typedef struct {
  AcKernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;


static TBConfig getOptimalTBConfig(const AcKernel kernel, const int3 dims, VertexBufferArray vba);

static __global__ void
flush_kernel(AcReal* arr, const size_t n, const AcReal value)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    arr[idx] = value;
}
static __global__ void
flush_kernel_int(int* arr, const size_t n, const int value)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    arr[idx] = value;
}
template <typename T>
T TO_CORRECT_ORDER(const T vol)
{
	return vol;
}
size_t TO_CORRECT_ORDER(const size_t size)
{
	return size;
}
#define KERNEL_LAUNCH(func,bgp,tpb,...) func<<<TO_CORRECT_ORDER(bpg),TO_CORRECT_ORDER(tpb),__VA_ARGS__>>>
AcResult
acKernelFlush(const cudaStream_t stream, AcReal* arr, const size_t n,
              const AcReal value)
{
  ERRCHK_ALWAYS(arr);
  const size_t tpb = 256;
  const size_t bpg = (size_t)(ceil((double)n / tpb));
  KERNEL_LAUNCH(flush_kernel,bpg,tpb,0,stream)(arr,n,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

AcResult
acKernelFlushInt(const cudaStream_t stream, int* arr, const size_t n,
              const int value)
{
  const size_t tpb = 256;
  const size_t bpg = (size_t)(ceil((double)n / tpb));
  KERNEL_LAUNCH(flush_kernel_int,bpg,tpb,0,stream)(arr,n,value);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

#if USE_COMPRESSIBLE_MEMORY
#include <cuda.h>

#define ERRCHK_CU_ALWAYS(x) ERRCHK_ALWAYS((x) == CUDA_SUCCESS)

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
#endif

AcResult
acPBAReset(const cudaStream_t stream, ProfileBufferArray* pba, const size3_t counts)
{
  // Set pba.in data to all-nan and pba.out to 0
  for (int i = 0; i < NUM_PROFILES; ++i) {
    acKernelFlush(stream, pba->in[i],  prof_count(Profile(i),counts), (AcReal)0);
    acKernelFlush(stream, pba->out[i], prof_count(Profile(i),counts), (AcReal)0);
  }
  return AC_SUCCESS;
}
void
device_malloc(void** dst, const int bytes)
{
 #if USE_COMPRESSIBLE_MEMORY 
    ERRCHK_CUDA_ALWAYS(mallocCompressible(dst, bytes));
 #else
    ERRCHK_CUDA_ALWAYS(cudaMalloc(dst, bytes));
  #endif
  ERRCHK_ALWAYS(dst != NULL);
}
void
device_malloc(AcReal** dst, const int bytes)
{
	device_malloc((void**)dst,bytes);
}

template <typename T>
void
device_free(T** dst, const int bytes)
{
#if USE_COMPRESSIBLE_MEMORY
  freeCompressible(*dst, bytes);
#else
  cudaFree(*dst);
  //used to silence unused warning
  (void)bytes;
#endif
  *dst = NULL;
}

ProfileBufferArray
acPBACreate(const size3_t counts)
{
  ProfileBufferArray pba{};
  pba.count = counts.z;
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),counts)*sizeof(AcReal);
    device_malloc(&pba.in[i],  bytes);
    device_malloc(&pba.out[i], bytes);
  }

  acPBAReset(0, &pba, counts);
  cudaDeviceSynchronize();
  return pba;
}

void
acPBADestroy(ProfileBufferArray* pba, const size3_t counts)
{
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),counts)*sizeof(AcReal);
    device_free(&pba->in[i],  bytes);
    device_free(&pba->out[i], bytes);
    pba->in[i]  = NULL;
    pba->out[i] = NULL;
  }
  pba->count = 0;
}

AcResult
acVBAReset(const cudaStream_t stream, VertexBufferArray* vba)
{
  const size_t count = vba->bytes / sizeof(vba->in[0][0]);

  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    ERRCHK_ALWAYS(vba->in[i]);
    ERRCHK_ALWAYS(vba->out[i]);
    acKernelFlush(stream, vba->in[i], count, (AcReal)0);
    acKernelFlush(stream, vba->out[i], count, (AcReal)0);
  }
  memset(&vba->kernel_input_params,0,sizeof(acKernelInputParams));
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBAReset(stream, &vba->profiles, (size3_t){vba->mx,vba->my,vba->mz});
  return AC_SUCCESS;
}


template <typename T>
void
device_malloc(T** dst, const int bytes)
{
 #if USE_COMPRESSIBLE_MEMORY 
    ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)dst, bytes));
 #else
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)dst, bytes));
  #endif
}

#include "memcpy_to_gmem_arrays.h"

#include "memcpy_from_gmem_arrays.h"

template <typename P>
struct allocate_arrays
{
	void operator()(const AcMeshInfoParams& config) 
	{
		for(P array : get_params<P>())
		{
			if (config[array] != nullptr && !is_dconst(array) && is_alive(array) && !has_const_dims(array))
			{

#if AC_VERBOSE
				fprintf(stderr,"Allocating %s|%d\n",get_name(array),get_array_length(array,config.scalars));
				fflush(stderr);
#endif
				auto d_mem_ptr = get_empty_pointer(array);
			        device_malloc(((void**)&d_mem_ptr), sizeof(config[array][0])*get_array_length(array,config.scalars));
				memcpy_to_gmem_array(array,d_mem_ptr);
			}
		}
	}
};

size3_t
buffer_dims(const AcMeshInfoParams config)
{
	auto mm = config[AC_mlocal];
	return (size3_t){as_size_t(mm.x),as_size_t(mm.y),as_size_t(mm.z)};
}

VertexBufferArray
acVBACreate(const AcMeshInfoParams config)
{
  //TP: !HACK!
  //TP: Get active dimensions at the time VBA is created, works for now but should be moved somewhere else
  dimension_inactive = config[AC_dimension_inactive];
  const size3_t counts = buffer_dims(config);
  VertexBufferArray vba;
  size_t count = counts.x*counts.y*counts.z;
  size_t bytes = sizeof(vba.in[0][0]) * count;
  vba.bytes          = bytes;
  vba.mx             = counts.x;
  vba.my             = counts.y;
  vba.mz             = counts.z;
#if AC_ADJACENT_VERTEX_BUFFERS
  const size_t allbytes = bytes*NUM_VTXBUF_HANDLES;
  AcReal *allbuf_in, *allbuf_out;

  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&allbuf_in, allbytes));
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&allbuf_out, allbytes));

  acKernelFlush(STREAM_DEFAULT,allbuf_in, count*NUM_VTXBUF_HANDLES, (AcReal)0.0);
  ERRCHK_CUDA_ALWAYS(cudaMemset((void*)allbuf_out, 0, allbytes));

  vba.in[0]=allbuf_in; vba.out[0]=allbuf_out;
printf("i,vbas[0]= %p %p \n",vba.in[0],vba.out[0]);
  for (size_t i = 1; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.in [i]=vba.in [i-1]+count;
    vba.out[i]=vba.out[i-1]+count;
printf("i,vbas[i]= %zu %p %p\n",i,vba.in[i],vba.out[i]);
  }
#else
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    device_malloc((void**) &vba.in[i],bytes);
    //Auxiliary fields need only a single copy so out can point to in
    if (vtxbuf_is_auxiliary[i])
    {
      vba.out[i] = vba.in[i];
    }else{
      device_malloc((void**) &vba.out[i],bytes);
    }
  }
#endif
  //Allocate workbuffers
  for (int i = 0; i < NUM_WORK_BUFFERS; ++i)
    device_malloc((void**)&vba.w[i],bytes);


  AcArrayTypes::run<allocate_arrays>(config);

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  vba.profiles = acPBACreate(counts);

  acVBAReset(0, &vba);
  cudaDeviceSynchronize();
  return vba;
}

template <typename P>
struct update_arrays
{
	void operator()(const AcMeshInfoParams& config)
	{
		for(P array : get_params<P>())
		{
			if (is_dconst(array) || !is_alive(array) || has_const_dims(array)) continue;
			auto config_array = config[array];
			auto gmem_array   = get_empty_pointer(array);
			memcpy_from_gmem_array(array,gmem_array);
			size_t bytes = sizeof(config_array[0])*get_array_length(array,config.scalars);
			if (config_array == nullptr && gmem_array != nullptr) 
				device_free(&gmem_array,bytes);
			else if (config_array != nullptr && gmem_array  == nullptr) 
				device_malloc(&gmem_array,bytes);
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};
void
acUpdateArrays(const AcMeshInfoParams config)
{
  AcArrayTypes::run<update_arrays>(config);
}

template <typename P>
struct free_arrays
{
	void operator()(const AcMeshInfoParams& config)
	{
		for(P array: get_params<P>())
		{
			auto config_array = config[array];
			if (config_array == nullptr || is_dconst(array) || !is_alive(array) || has_const_dims(array)) continue;
			auto gmem_array = get_empty_pointer(array);
			memcpy_from_gmem_array(array,gmem_array);
			device_free(&gmem_array, get_array_length(array,config.scalars));
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};

void
acVBADestroy(VertexBufferArray* vba, const AcMeshInfoParams config)
{
  for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) { 
    //TP: if dead then not allocated and thus nothing to free
    device_free(&(vba->in[i]), vba->bytes);
    if (vtxbuf_is_auxiliary[i])
      vba->out[i] = NULL;
    else
      device_free(&(vba->out[i]), vba->bytes);
  }
  //Free workbuffers 
  for (int i = 0; i < NUM_WORK_BUFFERS; ++i) 
    device_free(&(vba->w[i]), vba->bytes);

  //Free arrays
  AcArrayTypes::run<free_arrays>(config);
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBADestroy(&vba->profiles,(size3_t){vba->mx,vba->my,vba->mz});
  vba->bytes = 0;
  vba->mx    = 0;
  vba->my    = 0;
  vba->mz    = 0;
}



int
get_num_of_reduce_output(const dim3 bpg, const dim3 tpb)
{
	const size_t warp_size = get_warp_size();
	const int num_of_warps_per_block = (tpb.x*tpb.y*tpb.z + warp_size-1)/warp_size;
	const int num_of_blocks = bpg.x*bpg.y*bpg.z;
	return num_of_warps_per_block*num_of_blocks;
}

AcResult
acLaunchKernel(AcKernel kernel, const cudaStream_t stream, const int3 start,
               const int3 end, VertexBufferArray vba)
{
  const int3 n = end - start;

  const TBConfig tbconf = getOptimalTBConfig(kernel, n, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));

  const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));
  if (reduce_offsets[kernel].find(start) == reduce_offsets[kernel].end())
  {
  	reduce_offsets[kernel][start] = kernel_running_reduce_offsets[kernel];
  	kernel_running_reduce_offsets[kernel] += get_num_of_reduce_output(bpg,tpb);
  }

  vba.reduce_offset = reduce_offsets[kernel][start];
  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
  KERNEL_LAUNCH(kernels[kernel],bpg,tpb,smem,stream)(start,end,vba);
  ERRCHK_CUDA_KERNEL();

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}

AcResult
acBenchmarkKernel(AcKernel kernel, const int3 start, const int3 end,
                  VertexBufferArray vba)
{
  const int3 n = end - start;

  const TBConfig tbconf = getOptimalTBConfig(kernel, n, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));
  const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));

  // Timer create
  cudaEvent_t tstart, tstop;
  cudaEventCreate(&tstart);
  cudaEventCreate(&tstop);

  // Warmup
  cudaEventRecord(tstart);
  KERNEL_LAUNCH(kernels[kernel],bpg, tpb, smem)(start, end, vba);
  cudaEventRecord(tstop);
  cudaEventSynchronize(tstop);
  ERRCHK_CUDA_KERNEL();
  cudaDeviceSynchronize();

  // Benchmark
  cudaEventRecord(tstart); // Timing start
  KERNEL_LAUNCH(kernels[kernel],bpg,tpb,smem)(start, end, vba);
  cudaEventRecord(tstop); // Timing stop
  cudaEventSynchronize(tstop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, tstart, tstop);
  printf("Kernel %s time elapsed: %g ms\n", kernel_names[kernel],(double)milliseconds);

  // Timer destroy
  cudaEventDestroy(tstart);
  cudaEventDestroy(tstop);

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}


AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t /* stream */,
              const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  // Note important cudaDeviceSynchronize below
  //
  // Constant memory allocated for stencils is shared among kernel
  // invocations, therefore a race condition is possible when updating
  // the coefficients. To avoid this, all kernels that can access
  // the coefficients must be completed before starting async copy to
  // constant memory
  cudaDeviceSynchronize();

  const size_t bytes = sizeof(data[0][0][0]) * STENCIL_DEPTH * STENCIL_HEIGHT *
                       STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyToSymbol(
      stencils, data, bytes, stencil * bytes, cudaMemcpyHostToDevice);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};

AcResult
acLoadStencils(const cudaStream_t stream,
               const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                                [STENCIL_WIDTH])
{
  int retval = 0;
  for (size_t i = 0; i < NUM_STENCILS; ++i)
    retval |= acLoadStencil((Stencil)i, stream, data[i]);
  return (AcResult)retval;
}

AcResult
acStoreStencil(const Stencil stencil, const cudaStream_t /* stream */,
               AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  // Ensure all acLoadUniform calls have completed before continuing
  cudaDeviceSynchronize();

  const size_t bytes = sizeof(data[0][0][0]) * STENCIL_DEPTH * STENCIL_HEIGHT *
                       STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyFromSymbol(
      data, stencils, bytes, stencil * bytes, cudaMemcpyDeviceToHost);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};


template <typename P, typename V>
static AcResult
acLoadUniform(const P param, const V value)
{
	if constexpr (std::is_same<P,AcReal>::value)
	{
  		if (isnan(value)) {
  		  fprintf(stderr,
  		          "WARNING: Passed an invalid value %g to device constant %s. "
  		          "Skipping.\n",
  		          (double)value, realparam_names[param]);
  		  return AC_FAILURE;
  		}
	}
	else if constexpr (std::is_same<P,AcReal3>::value)
	{
  		if (isnan(value.x) || isnan(value.y) || isnan(value.z)) {
  		  fprintf(stderr,
  		          "WARNING: Passed an invalid value (%g, %g, %g) to device constant "
  		          "%s. Skipping.\n",
  		          (double)value.x, (double)value.y, (double)value.z,
  		          real3param_names[param]);
  		  return AC_FAILURE;
  		}
	}
  	ERRCHK_ALWAYS(param < get_num_params<P>());
  	cudaDeviceSynchronize(); /* See note in acLoadStencil */

  	const size_t offset =  get_address(param) - (size_t)&d_mesh_info;
  	const cudaError_t retval = cudaMemcpyToSymbol(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice);
  	return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
}



template <typename P, typename V>
static AcResult
acLoadArrayUniform(const P array, const V* values, const size_t length)
{
#if AC_VERBOSE
	fprintf(stderr,"Loading %s\n",get_name(array));
	fflush(stderr);
#endif
	cudaDeviceSynchronize();
	ERRCHK_ALWAYS(values  != nullptr);
	const size_t bytes = length*sizeof(values[0]);
	if (!is_dconst(array))
	{
		if (!is_alive(array)) return AC_NOT_ALLOCATED;
		if (has_const_dims(array))
		{
			memcpy_to_const_dims_gmem_array(array,values);
			return AC_SUCCESS;
		}
		auto dst_ptr = get_empty_pointer(array);
		memcpy_from_gmem_array(array,dst_ptr);
		ERRCHK_ALWAYS(dst_ptr != nullptr);
		if (dst_ptr == nullptr)
		{
			fprintf(stderr,"FATAL AC ERROR from acLoadArrayUniform\n");
			exit(EXIT_FAILURE);
		}
#if AC_VERBOSE
		fprintf(stderr,"Calling (cuda/hip)memcpy %s|%ld\n",get_name(array),length);
		fflush(stderr);
#endif
		ERRCHK_CUDA_ALWAYS(cudaMemcpy(dst_ptr,values,bytes,cudaMemcpyHostToDevice));
	}
	else 
		ERRCHK_CUDA_ALWAYS(load_array(values, bytes, array));
#if AC_VERBOSE
	fprintf(stderr,"Loaded %s\n",get_name(array));
	fflush(stderr);
#endif
	return AC_SUCCESS;
}

template <typename P, typename V>
AcResult
acStoreUniform(const P param, V* value)
{
	ERRCHK_ALWAYS(param < get_num_params<P>());
	cudaDeviceSynchronize();
  	const size_t offset =  get_address(param) - (size_t)&d_mesh_info;
	const cudaError_t retval = cudaMemcpyFromSymbol(value, d_mesh_info, sizeof(V), offset, cudaMemcpyDeviceToHost);
	return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
}

template <typename P, typename V>
AcResult
acStoreArrayUniform(const P array, V* values, const size_t length)
{
	ERRCHK_ALWAYS(values  != nullptr);
	const size_t bytes = length*sizeof(values[0]);
	if (!is_dconst(array))
	{
		if (!is_alive(array)) return AC_NOT_ALLOCATED;
		if (has_const_dims(array))
		{
			memcpy_from_gmem_array(array,values);
			return AC_SUCCESS;
		}
		auto src_ptr = get_empty_pointer(array);
		memcpy_from_gmem_array(array,src_ptr);
		ERRCHK_ALWAYS(src_ptr != nullptr);
		ERRCHK_CUDA_ALWAYS(cudaMemcpy(values, src_ptr, bytes, cudaMemcpyDeviceToHost));
	}
	else
		ERRCHK_CUDA_ALWAYS(store_array(values, bytes, array));
	return AC_SUCCESS;
}

#include "load_and_store_uniform_funcs.h"


//TP: best would be to use carriage return to have a single line that simple keeps growing but that seems not to be always supported in SLURM environments. 
// Or at least requires actions from the user
void printProgressBar(FILE* stream, const int progress) {
    int barWidth = 50;
    fprintf(stream,"[");  // Start a new line
    int pos = barWidth * progress / 100;

    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            fprintf(stream,COLORIZE("=",GREEN));  
        } else if (i == pos) {
            fprintf(stream,COLORIZE(">",YELLOW));  
        } else {
            fprintf(stream," ");
        }
    }
    fprintf(stream,"] %d %%", progress);
}
void
logAutotuningStatus(const size_t counter, const size_t num_samples, const AcKernel kernel)
{
    const size_t percent_of_num_samples = num_samples/100;
    for (size_t progress = 0; progress <= 100; ++progress)
    {
	      if (counter == percent_of_num_samples*progress  && grid_pid == 0 && (progress % 10 == 0))
	      {
    			fprintf(stderr,"\nAutotuning %s ",kernel_names[kernel]);
    			printProgressBar(stderr,progress);
			if (progress == 100) fprintf(stderr,"\n");
			fflush(stderr);
	      }
    }
}

static AcAutotuneMeasurement
gather_best_measurement(const AcAutotuneMeasurement local_best)
{
#if AC_MPI_ENABLED
	return gather_func(local_best);
#else
        return local_best;
#endif
}

void
make_vtxbuf_input_params_safe(VertexBufferArray& vba, const AcKernel kernel)
{
  //TP: have to set reduce offset zero since it might not be
  vba.reduce_offset = 0;
#include "safe_vtxbuf_input_params.h"
}

static TBConfig
autotune(const AcKernel kernel, const int3 dims, VertexBufferArray vba)
{
  make_vtxbuf_input_params_safe(vba,kernel);
  // printf("Autotuning kernel '%s' (%p), block (%d, %d, %d), implementation "
  //        "(%d):\n",
  //        kernel_names[id], kernel, dims.x, dims.y, dims.z, IMPLEMENTATION);
  // fflush(stdout);

#if 0
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed
#endif

  TBConfig c = {
      .kernel = kernel,
      .dims   = dims,
      .tpb    = (dim3){0, 0, 0},
  };

  const int3 ghosts = (int3){
	  dimension_inactive.x ? 0 : NGHOST,
	  dimension_inactive.y ? 0 : NGHOST,
	  dimension_inactive.z ? 0 : NGHOST
  };
  const int3 start = ghosts;
  const int3 end = start + dims;


  //TP: since autotuning should be quite fast when the dim is not NGHOST only log for actually 3d portions
  const bool builtin_kernel = strlen(kernel_names[kernel]) > 2 && kernel_names[kernel][0] == 'A' && kernel_names[kernel][1] == 'C';
  const bool large_launch = (dims.x > ghosts.x && dims.y > ghosts.y && dims.z > ghosts.z);
  const bool log = !builtin_kernel && large_launch;

  dim3 best_tpb(0, 0, 0);
  float best_time     = INFINITY;
  const int num_iters = 2;

  // Get device hardware information
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  const int max_threads_per_block = MAX_THREADS_PER_BLOCK
                                        ? min(props.maxThreadsPerBlock,
                                              MAX_THREADS_PER_BLOCK)
                                        : props.maxThreadsPerBlock;
  const size_t max_smem           = props.sharedMemPerBlock;

  // Old heuristic
  // for (int z = 1; z <= max_threads_per_block; ++z) {
  //   for (int y = 1; y <= max_threads_per_block; ++y) {
  //     for (int x = max(y, z); x <= max_threads_per_block; ++x) {

  // New: require that tpb.x is a multiple of the minimum transaction or L2
  // cache line size
  const int minimum_transaction_size_in_elems = 32 / sizeof(AcReal);
  // New: restrict tpb.x to be at most dims.x since launching threads that are known to be oob feels simply wasteful
  const int x_increment = min(
		  			minimum_transaction_size_in_elems,
		  			dims.x
		            );

  std::vector<int3> samples{};
  for (int z = 1; z <= min(max_threads_per_block,dims.z); ++z) {
    for (int y = 1; y <= min(max_threads_per_block,dims.y); ++y) {
      for (int x = x_increment;
           x <= min(max_threads_per_block,dims.x); x += x_increment) {


        if (x * y * z > max_threads_per_block)
          break;
        const dim3 tpb(x, y, z);
        const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        if (smem > max_smem)
          continue;

        if ((x * y * z) % props.warpSize && (x*y*z) >props.warpSize)
          continue;

        if (!is_valid_configuration(to_volume(dims), to_volume(tpb),kernel))
          continue;
	//TP: should be emplace back but on my laptop the CUDA compiler gives a cryptic error message that I do not care to debug
        samples.push_back((int3){x,y,z});
      }
    }
  }
  size_t counter  = 0;
  size_t start_samples{};
  size_t end_samples{};
  if(large_launch && AC_MPI_ENABLED)
  {
  	const size_t portion = (size_t)ceil((1.0*samples.size())/nprocs);
  	start_samples = portion*grid_pid;
  	end_samples   = min(samples.size(), portion*(grid_pid+1));
  }
  else
  {
  	start_samples = 0;
  	end_samples   = samples.size();
  }
  const size_t n_samples = end_samples-start_samples;
  const Kernel func = kernels[kernel];
  for(size_t sample  = start_samples; sample < end_samples; ++sample)
  {
        if (log) logAutotuningStatus(counter,n_samples,kernel);
        ++counter;
        auto x = samples[sample].x;
        auto y = samples[sample].y;
        auto z = samples[sample].z;
        const dim3 tpb(x, y, z);
        const dim3 bpg    = to_dim3(
                                get_bpg(to_volume(dims),
                                to_volume(tpb))
                                );
        const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        cudaEvent_t tstart, tstop;
        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        KERNEL_LAUNCH(func,bpg, tpb, smem)(start, end, vba); // Dryrun
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < num_iters; ++i)
          KERNEL_LAUNCH(func,bpg, tpb, smem)(start, end, vba); // Dryrun
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
	  //TP: reset autotune results
          FILE* fp = fopen(autotune_csv_path,"w");
	  fclose(fp);
          // Exit in case of unrecoverable error that needs a device reset
          ERRCHK_CUDA_KERNEL_ALWAYS();
          ERRCHK_CUDA_ALWAYS(cudaGetLastError());
          continue;
        }

        if (milliseconds < best_time) {
          best_time = milliseconds;
          best_tpb  = tpb;
        }

        // printf("Auto-optimizing... Current tpb: (%d, %d, %d), time %f ms\n",
        //        tpb.x, tpb.y, tpb.z, (double)milliseconds / num_iters);
        // fflush(stdout);
  }
  const AcAutotuneMeasurement best_measurement = 
	  large_launch ? gather_best_measurement({best_time,best_tpb}) : (AcAutotuneMeasurement){best_time,best_tpb};
  c.tpb = best_measurement.tpb;
  best_time = best_measurement.time;
  if(grid_pid == 0)
  {
        FILE* fp = fopen(autotune_csv_path, "a");
        ERRCHK_ALWAYS(fp);
#if IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
        fprintf(fp, "%d, (%d, %d, %d), (%d, %d, %d), %g\n", IMPLEMENTATION, nx, ny,
                nz, best_tpb.x, best_tpb.y, best_tpb.z,
                (double)best_time / num_iters);
#else
        fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %d, %g, %s\n", IMPLEMENTATION, kernel, dims.x,
                dims.y, dims.z, best_tpb.x, best_tpb.y, best_tpb.z,
                (double)best_time / num_iters, kernel_names[kernel]);
#endif
        fclose(fp);
  }
  if (c.tpb.x * c.tpb.y * c.tpb.z <= 0) {
    fprintf(stderr,
            "Fatal error: failed to find valid thread block dimensions for (%d,%d,%d) launch.\n"
            ,dims.x,dims.y,dims.z);
  }
  ERRCHK_ALWAYS(c.tpb.x * c.tpb.y * c.tpb.z > 0);
  return c;
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}
static int3
read_optim_tpb(const AcKernel kernel, const int3 dims)
{
  if(!file_exists(autotune_csv_path)) return {-1,-1,-1};
  const char* filename = autotune_csv_path;
  FILE *file = fopen ( filename, "r" );
  int3 res = {-1,-1,-1};
  double best_time     = (double)INFINITY;
  string_vec entries[1000];
  memset(entries,0,sizeof(string_vec)*1000);
  const int n_entries = get_csv_entries(entries,file);
  for(int i = 0; i < n_entries; ++i)
  {
	  string_vec entry = entries[i];
	  if(entry.size > 1)
      	  {
      	     int kernel_index  = atoi(entry.data[1]);
      	     int3 read_dims = {atoi(entry.data[2]), atoi(entry.data[3]), atoi(entry.data[4])};
      	     int3 tpb = {atoi(entry.data[5]), atoi(entry.data[6]), atoi(entry.data[7])};
      	     double time = atof(entry.data[8]);
      	     if(time < best_time && kernel_index == kernel && read_dims == dims)
      	     {
      	    	 best_time = time;
      	    	 res       = tpb;
      	     }
      	  }
      	  for(size_t elem = 0; elem < entry.size; ++elem)
      	         free((char*)entry.data[elem]);
      	  free_str_vec(&entry);
  }
  fclose(file);
  return res;
}


static TBConfig
getOptimalTBConfig(const AcKernel kernel, const int3 dims, VertexBufferArray vba)
{
  for (auto c : tbconfigs)
    if (c.kernel == kernel && c.dims == dims)
      return c;

  const int3 read_tpb = read_optim_tpb(kernel,dims);
  TBConfig c  = (read_tpb != (int3){-1,-1,-1})
          ? (TBConfig){kernel,dims,(dim3){(uint32_t)read_tpb.x, (uint32_t)read_tpb.y, (uint32_t)read_tpb.z}}
          : autotune(kernel,dims,vba);
  tbconfigs.push_back(c);
  return c;
}

AcKernel
acGetOptimizedKernel(const AcKernel kernel_enum, const VertexBufferArray vba)
{
	//#include "user_kernel_ifs.h"
	//silence unused warnings
	(void)vba;
	//TP: for now this is no-op in the future in some cases we choose which kernel to call based on the input params
	return kernel_enum;
	//return kernels[(int) kernel_enum];
}
void
acVBASwapBuffer(const Field field, VertexBufferArray* vba)
{
  AcReal* tmp     = vba->in[field];
  vba->in[field]  = vba->out[field];
  vba->out[field] = tmp;
}

void
acVBASwapBuffers(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_FIELDS; ++i)
    acVBASwapBuffer((Field)i, vba);
}

void
acPBASwapBuffer(const Profile profile, VertexBufferArray* vba)
{
  AcReal* tmp                = vba->profiles.in[profile];
  vba->profiles.in[profile]  = vba->profiles.out[profile];
  vba->profiles.out[profile] = tmp;
}

void
acPBASwapBuffers(VertexBufferArray* vba)
{
  for (int i = 0; i < NUM_PROFILES; ++i)
    acPBASwapBuffer((Profile)i, vba);
}

AcResult
acLoadMeshInfo(const AcMeshInfoScalars info, const cudaStream_t stream)
{
  for (int i = 0; i < NUM_INT_PARAMS; ++i)
    ERRCHK_ALWAYS(acLoadIntUniform(stream, (AcIntParam)i, info.int_params[i]) ==
                  AC_SUCCESS);

  for (int i = 0; i < NUM_INT3_PARAMS; ++i)
    ERRCHK_ALWAYS(acLoadInt3Uniform(stream, (AcInt3Param)i,
                                    info.int3_params[i]) == AC_SUCCESS);

  for (int i = 0; i < NUM_REAL_PARAMS; ++i)
    ERRCHK_ALWAYS(acLoadRealUniform(stream, (AcRealParam)i,
                                    info.real_params[i]) == AC_SUCCESS);

  for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
    ERRCHK_ALWAYS(acLoadReal3Uniform(stream, (AcReal3Param)i,
                                     info.real3_params[i]) == AC_SUCCESS);

  return AC_SUCCESS;
}

//---------------
// static __host__ __device__ constexpr size_t
// acShapeSize(const AcShape& shape)
size_t
acShapeSize(const AcShape shape)
{
  return shape.x * shape.y * shape.z * shape.w;
}

static __host__ __device__ constexpr bool
acOutOfBounds(const AcIndex& index, const AcShape& shape)
{
  return (index.x >= shape.x) || //
         (index.y >= shape.y) || //
         (index.z >= shape.z) || //
         (index.w >= shape.w);
}

static __host__ __device__ constexpr AcIndex __attribute__((unused))
min(const AcIndex& a, const AcIndex& b)
{
  return (AcIndex){
      a.x < b.x ? a.x : b.x,
      a.y < b.y ? a.y : b.y,
      a.z < b.z ? a.z : b.z,
      a.w < b.w ? a.w : b.w,
  };
}

static __host__ __device__ constexpr AcIndex
operator+(const AcIndex& a, const AcIndex& b)
{
  return (AcIndex){
      a.x + b.x,
      a.y + b.y,
      a.z + b.z,
      a.w + b.w,
  };
}

static __host__ __device__ constexpr AcIndex __attribute__((unused))
operator-(const AcIndex& a, const AcIndex& b) 
{
  return (AcIndex){
      a.x - b.x,
      a.y - b.y,
      a.z - b.z,
      a.w - b.w,
  };
}

static __host__ __device__ constexpr AcIndex
to_spatial(const size_t i, const AcShape& shape)
{
  return (AcIndex){
      .x = i % shape.x,
      .y = (i / shape.x) % shape.y,
      .z = (i / (shape.x * shape.y)) % shape.z,
      .w = i / (shape.x * shape.y * shape.z),
  };
}

static __host__ __device__ constexpr size_t
to_linear(const AcIndex& index, const AcShape& shape)
{
  return index.x +           //
         index.y * shape.x + //
         index.z * shape.x * shape.y + index.w * shape.x * shape.y * shape.z;
}

static __global__ void
reindex(const AcReal* in, const AcIndex in_offset, const AcShape in_shape,
        AcReal* out, const AcIndex out_offset, const AcShape out_shape,
        const AcShape block_shape)
{
  const size_t i    = (size_t)threadIdx.x + blockIdx.x * blockDim.x;
  const AcIndex idx = to_spatial(i, block_shape);

  const AcIndex in_pos  = idx + in_offset;
  const AcIndex out_pos = idx + out_offset;

  if (acOutOfBounds(idx, block_shape) || //
      acOutOfBounds(in_pos, in_shape) || //
      acOutOfBounds(out_pos, out_shape))
    return;

  const size_t in_idx  = to_linear(in_pos, in_shape);
  const size_t out_idx = to_linear(out_pos, out_shape);

  out[out_idx] = in[in_idx];
}

AcResult
acReindex(const cudaStream_t stream, //
          const AcReal* in, const AcIndex in_offset, const AcShape in_shape,
          AcReal* out, const AcIndex out_offset, const AcShape out_shape,
          const AcShape block_shape)
{
  const size_t count = acShapeSize(block_shape);
  const size_t tpb   = min(256ul, count);
  const size_t bpg   = (count + tpb - 1) / tpb;

  KERNEL_LAUNCH(reindex,bpg, tpb, 0, stream)(in, in_offset, in_shape, //
                                   out, out_offset, out_shape, block_shape);
  ERRCHK_CUDA_KERNEL();

  return AC_SUCCESS;
}

typedef struct {
  AcReal *x, *y, *z;
} SOAVector;

typedef struct {
  // Input vectors
  SOAVector A[1];
  size_t A_count;
  SOAVector B[4];
  size_t B_count;
  // Note: more efficient with A_count < B_count

  // Output vectors
  SOAVector C[1 * 4];
  // C count = A_count*B_count
} CrossProductArrays;

static __global__ void UNUSED
reindex_cross(const CrossProductArrays arrays, const AcIndex in_offset,
              const AcShape in_shape, const AcIndex out_offset,
              const AcShape out_shape, const AcShape block_shape)
{
  const AcIndex idx = to_spatial((size_t)threadIdx.x + blockIdx.x * blockDim.x
		  , block_shape);

  const AcIndex in_pos  = idx + in_offset;
  const AcIndex out_pos = idx + out_offset;

  if (acOutOfBounds(idx, block_shape) || //
      acOutOfBounds(in_pos, in_shape) || //
      acOutOfBounds(out_pos, out_shape))
    return;

  const size_t in_idx  = to_linear(in_pos, in_shape);
  const size_t out_idx = to_linear(out_pos, out_shape);

  for (size_t j = 0; j < arrays.A_count; ++j) {
    const AcReal3 a = {
        arrays.A[j].x[in_idx],
        arrays.A[j].y[in_idx],
        arrays.A[j].z[in_idx],
    };
    for (size_t i = 0; i < arrays.B_count; ++i) {
      const AcReal3 b = {
          arrays.B[i].x[in_idx],
          arrays.B[i].y[in_idx],
          arrays.B[i].z[in_idx],
      };
      const AcReal3 res                           = AC_cross(a, b);
      arrays.C[i + j * arrays.B_count].x[out_idx] = res.x;
      arrays.C[i + j * arrays.B_count].y[out_idx] = res.y;
      arrays.C[i + j * arrays.B_count].z[out_idx] = res.z;
    }
  }
}

#if 0
__global__ void
map_cross_product(const CrossProductInputs inputs, const AcIndex start,
                  const AcIndex end)
{

  const AcIndex tid = {
      .x = threadIdx.x + blockIdx.x * blockDim.x,
      .y = threadIdx.y + blockIdx.y * blockDim.y,
      .z = threadIdx.z + blockIdx.z * blockDim.z,
      .w = 0,
  };

  const AcIndex in_idx3d = start + tid;
  const size_t in_idx = DEVICE_VTXBUF_IDX(in_idx3d.x, in_idx3d.y, in_idx3d.z);

  const AcShape dims   = end - start;
  const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;

  const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y &&
                             in_idx3d.z < end.z;
  if (within_bounds) {
    for (size_t i = 0; i < inputs.A_count; ++i) {
      const AcReal3 a = (AcReal3){
          inputs.A[i].x[in_idx],
          inputs.A[i].y[in_idx],
          inputs.A[i].z[in_idx],
      };
      for (size_t j = 0; j < inputs.B_count; ++j) {
        const AcReal3 b = (AcReal3){
            inputs.B[j].x[in_idx],
            inputs.B[j].y[in_idx],
            inputs.B[j].z[in_idx],
        };
        const AcReal3 res            = cross(a, b);
        inputs.outputs[j].x[out_idx] = res.x;
        inputs.outputs[j].y[out_idx] = res.y;
        inputs.outputs[j].z[out_idx] = res.z;
      }
    }
  }
}
#endif

#ifdef AC_TFM_ENABLED
AcResult
acReindexCross(const cudaStream_t stream, //
               const VertexBufferArray vba, const AcIndex in_offset,
               const AcShape in_shape, //
               AcReal* out, const AcIndex out_offset, const AcShape out_shape,
               const AcShape block_shape)
{
  const SOAVector uu = {
      .x = vba.in[VTXBUF_UUX],
      .y = vba.in[VTXBUF_UUY],
      .z = vba.in[VTXBUF_UUZ],
  };
  const SOAVector bb11 = {
      .x = vba.in[TF_b11_x],
      .y = vba.in[TF_b11_y],
      .z = vba.in[TF_b11_z],
  };
  const SOAVector bb12 = {
      .x = vba.in[TF_b12_x],
      .y = vba.in[TF_b12_y],
      .z = vba.in[TF_b12_z],
  };
  const SOAVector bb21 = {
      .x = vba.in[TF_b21_x],
      .y = vba.in[TF_b21_y],
      .z = vba.in[TF_b21_z],
  };
  const SOAVector bb22 = {
      .x = vba.in[TF_b22_x],
      .y = vba.in[TF_b22_y],
      .z = vba.in[TF_b22_z],
  };

  const size_t block_offset = out_shape.x * out_shape.y * out_shape.z;
  const SOAVector out_bb11  = {
       .x = &out[3 * block_offset],
       .y = &out[4 * block_offset],
       .z = &out[5 * block_offset],
  };
  const SOAVector out_bb12 = {
      .x = &out[6 * block_offset],
      .y = &out[7 * block_offset],
      .z = &out[8 * block_offset],
  };
  const SOAVector out_bb21 = {
      .x = &out[9 * block_offset],
      .y = &out[10 * block_offset],
      .z = &out[11 * block_offset],
  };
  const SOAVector out_bb22 = {
      .x = &out[12 * block_offset],
      .y = &out[13 * block_offset],
      .z = &out[14 * block_offset],
  };

  const CrossProductArrays arrays = {
      .A       = {uu},
      .A_count = 1,
      .B       = {bb11, bb12, bb21, bb22},
      .B_count = 4,
      .C       = {out_bb11, out_bb12, out_bb21, out_bb22},
  };

  const size_t count = acShapeSize(block_shape);
  const size_t tpb   = min(256ul, count);
  const size_t bpg   = (count + tpb - 1) / tpb;

  KERNEL_LAUNCH(reindex_cross,bpg, tpb, 0, stream)(arrays, in_offset, in_shape,
                                         out_offset, out_shape, block_shape);
  return AC_SUCCESS;
}
#else
AcResult
acReindexCross(const cudaStream_t , //
               const VertexBufferArray , const AcIndex ,
               const AcShape , //
               AcReal* , const AcIndex , const AcShape ,
               const AcShape )
{
  ERROR("acReindexCross called but AC_TFM_ENABLED was false");
  return AC_FAILURE;
}
#endif

#if AC_USE_HIP
#include <hipcub/hipcub.hpp>
#define cub hipcub
#else
#include <cub/cub.cuh>
#endif

AcResult
acSegmentedReduce(const cudaStream_t stream, const AcReal* d_in,
                  const size_t count, const size_t num_segments, AcReal* d_out)
{
  size_t* offsets = (size_t*)malloc(sizeof(offsets[0]) * (num_segments + 1));
  ERRCHK_ALWAYS(offsets);
  for (size_t i = 0; i <= num_segments; ++i) {
    offsets[i] = i * (count / num_segments);
  }

  size_t* d_offsets = NULL;
  cudaMalloc(&d_offsets, sizeof(d_offsets[0]) * (num_segments + 1));
  ERRCHK_ALWAYS(d_offsets);
  cudaMemcpy(d_offsets, offsets, sizeof(d_offsets[0]) * (num_segments + 1),
             cudaMemcpyHostToDevice);

  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_segments, d_offsets, d_offsets + 1,
                                  stream);
  // printf("Temp storage: %zu bytes\n", temp_storage_bytes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  ERRCHK_ALWAYS(d_temp_storage);

  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                            d_out, num_segments, d_offsets, d_offsets + 1,
                            stream);

  cudaStreamSynchronize(
      stream); // Note, would not be needed if allocated at initialization
  cudaFree(d_temp_storage);
  cudaFree(d_offsets);
  free(offsets);
  return AC_SUCCESS;
}
typedef struct
{
	void* data;
	size_t bytes;
} AcDeviceTmpBuffer;
template <typename T>
void
cub_reduce(AcDeviceTmpBuffer& temp_storage, const cudaStream_t stream, const T* d_in, const size_t count, T* d_out,  AcReduceOp reduce_op)
{
  switch(reduce_op)
  {
	  case(REDUCE_SUM):
	  	cub::DeviceReduce::Sum(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream);
	  	break;
	  case(REDUCE_MIN):
	  	cub::DeviceReduce::Min(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream);
	  	break;
	  case(REDUCE_MAX):
	  	cub::DeviceReduce::Max(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream);
	  	break;
	default:
		ERRCHK_ALWAYS(reduce_op != NO_REDUCE);
  }
  if (cudaGetLastError() != cudaSuccess) {
          ERRCHK_CUDA_KERNEL_ALWAYS();
          ERRCHK_CUDA_ALWAYS(cudaGetLastError());
  }
}
template <typename T>
AcResult
acReduceBase(const cudaStream_t stream, const T* d_in, const size_t count, T* d_out, const AcReduceOp reduce_op)
{
  ERRCHK_ALWAYS(count != 0);
  ERRCHK_ALWAYS(d_in  != NULL);
  ERRCHK_ALWAYS(d_out != NULL);

  AcDeviceTmpBuffer temp_storage{NULL,0};
  cub_reduce(temp_storage,stream,d_in,count,d_out,reduce_op);

  ERRCHK_ALWAYS(temp_storage.bytes != 0);
  ERRCHK_CUDA_ALWAYS(cudaMalloc(&temp_storage.data, temp_storage.bytes));
  ERRCHK_ALWAYS(temp_storage.data);

  cub_reduce(temp_storage,stream,d_in,count,d_out,reduce_op);
  cudaStreamSynchronize(
    stream); // Note, would not be needed if allocated at initialization
  cudaFree(temp_storage.data);
  return AC_SUCCESS;
}

AcResult
acReduce(const cudaStream_t stream, const AcReal* d_in, const size_t count, AcReal* d_out, const AcReduceOp reduce_op)
{
	return acReduceBase(stream,d_in,count,d_out,reduce_op);
}


AcResult
acReduceInt(const cudaStream_t stream, const int* d_in, const size_t count, int* d_out, const AcReduceOp reduce_op)
{
	return acReduceBase(stream,d_in,count,d_out,reduce_op);
}

static __global__ void
multiply_inplace(const AcReal value, const size_t count, AcReal* array)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count)
    array[idx] *= value;
}

AcResult
acMultiplyInplace(const AcReal value, const size_t count, AcReal* array)
{
  const size_t tpb = 256;
  const size_t bpg = (count + tpb - 1) / tpb;
  KERNEL_LAUNCH(multiply_inplace,bpg, tpb,0,0)(value, count, array);
  ERRCHK_CUDA_KERNEL();
  return AC_SUCCESS;
}
#define TILE_DIM (32)

void __global__ 
transpose_xyz_to_zyx(const AcReal* src, AcReal* dst)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y,
		blockIdx.z*TILE_DIM
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + block_offset.x,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		threadIdx.x + block_offset.z,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.x
	};
	const bool in_oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	const bool out_oob =  (int)out_vertexIdx.x >= VAL(AC_mlocal).z ||  (int)out_vertexIdx.y >= VAL(AC_mlocal).y || (int)out_vertexIdx.z >= VAL(AC_mlocal).x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +VAL(AC_mlocal).z*out_vertexIdx.y + VAL(AC_mlocal_products).yz*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.z];
}
void __global__ 
transpose_xyz_to_zxy(const AcReal* src, AcReal* dst)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y,
		blockIdx.z*TILE_DIM
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + block_offset.x,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		threadIdx.x + block_offset.z,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.x
	};
	const bool in_oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	const bool out_oob =  (int)out_vertexIdx.x >= VAL(AC_mlocal).z ||  (int)out_vertexIdx.y >= VAL(AC_mlocal).y || (int)out_vertexIdx.z >= VAL(AC_mlocal).x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +VAL(AC_mlocal).z*out_vertexIdx.z + VAL(AC_mlocal_products).xz*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.z];
}
void __global__ 
transpose_xyz_to_xyz(const AcReal* src, AcReal* dst)
{
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y*TILE_DIM,
		blockIdx.z
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + block_offset.x,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.z
	};
	const bool oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	if(oob) return;
	dst[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)] = src[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)];
}
void __global__ 
transpose_xyz_to_yxz(const AcReal* src, AcReal* dst)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y*TILE_DIM,
		blockIdx.z
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + block_offset.x,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		threadIdx.x + block_offset.y,
		threadIdx.y + block_offset.x,
		threadIdx.z + block_offset.z
	};
	const bool in_oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	const bool out_oob =  (int)out_vertexIdx.x >= VAL(AC_mlocal).y ||  (int)out_vertexIdx.y >= VAL(AC_mlocal).x || (int)out_vertexIdx.z >= VAL(AC_mlocal).z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +VAL(AC_mlocal).y*out_vertexIdx.y + VAL(AC_mlocal_products).xy*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.y];
}
void __global__ 
transpose_xyz_to_yzx(const AcReal* src, AcReal* dst)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y*TILE_DIM,
		blockIdx.z
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + block_offset.x,
		threadIdx.y + block_offset.y,
		threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		threadIdx.x + block_offset.y,
		threadIdx.y + block_offset.x,
		threadIdx.z + block_offset.z
	};
	const bool in_oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	const bool out_oob =  (int)out_vertexIdx.x >= VAL(AC_mlocal).y ||  (int)out_vertexIdx.y >= VAL(AC_mlocal).x || (int)out_vertexIdx.z >= VAL(AC_mlocal).z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[DEVICE_VTXBUF_IDX(vertexIdx.x,vertexIdx.y,vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +VAL(AC_mlocal).y*out_vertexIdx.z + VAL(AC_mlocal_products).yz*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.y];
}
void __global__ 
transpose_xyz_to_xzy(const AcReal* src, AcReal* dst)
{
	const dim3 in_block_offset =
	{
		blockIdx.x*blockDim.x,
		blockIdx.y*blockDim.y,
		blockIdx.z*blockDim.z
	};

	const dim3 vertexIdx = 
	{
		threadIdx.x + in_block_offset.x,
		threadIdx.y + in_block_offset.y,
		threadIdx.z + in_block_offset.z
	};

	const bool oob  =  (int)vertexIdx.x  >= VAL(AC_mlocal).x    ||  (int)vertexIdx.y >= VAL(AC_mlocal).y     || (int)vertexIdx.z >= VAL(AC_mlocal).z;
	if(oob) return;
	dst[vertexIdx.x + VAL(AC_mlocal).x*vertexIdx.z + VAL(AC_mlocal_products).xz*vertexIdx.y] 
		= src[DEVICE_VTXBUF_IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)];
}
static AcResult
acTransposeXYZ_ZYX(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,1,32};

	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zyx,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_ZXY(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,1,32};

	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zxy,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_YXZ(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};

	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yxz,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_YZX(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};

	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yzx,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_XZY(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};
	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_xzy,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_XYZ(const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};
	const dim3 bpg = to_dim3(get_bpg(to_volume(dims),to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_xyz,bpg, tpb, 0, stream)(src,dst);
	return AC_SUCCESS;
}
AcResult
acTranspose(const AcMeshOrder order, const AcReal* src, AcReal* dst, const int3 dims, const cudaStream_t stream)
{
	switch(order)
	{
		case(XYZ):
			return acTransposeXYZ_XYZ(src,dst,dims,stream);
		case (XZY):
			return acTransposeXYZ_XZY(src,dst,dims,stream);
		case (YXZ):
			return acTransposeXYZ_YXZ(src,dst,dims,stream);
		case (YZX):
			return acTransposeXYZ_YZX(src,dst,dims,stream);
		case(ZXY):
			return acTransposeXYZ_ZXY(src,dst,dims,stream);
		case(ZYX):
			return acTransposeXYZ_ZYX(src,dst,dims,stream);
	}
	return AC_SUCCESS;
}
#include "load_ac_kernel_params.h"
