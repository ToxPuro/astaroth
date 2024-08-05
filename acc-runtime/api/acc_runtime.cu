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
#include "acc_runtime.h"

#include <math.h> 
#include <vector> // tbconfig

#include "errchk.h"
#include "math_utils.h"
#include <unordered_map>
#include <utility>

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#include <rocprim/rocprim.hpp>
#endif


#define USE_COMPRESSIBLE_MEMORY (0)
typedef struct Field3
{
	VertexBufferHandle x;
	VertexBufferHandle y;
	VertexBufferHandle z;
} Field3;



HOST_DEVICE_INLINE Field3 
MakeField3(const Field& x, const Field& y, const Field& z)
{
	return (Field3){x,y,z};
}
template <size_t N>
HOST_DEVICE_INLINE
std::array<Field3,N>
MakeField3(const Field (&x)[N], const Field (&y)[N], const Field (&z)[N])
{
	std::array<int3,N> res{};
	for(size_t i = 0; i < N; ++i)
		res[i] = (Field3){x,y,z};
	return res;
}

#include "acc/implementation.h"
#include "user_constants.h"

static dim3 last_tpb = (dim3){0, 0, 0};
static AcReal3 AC_INTERNAL_global_real_vec = {0.0,0.0,0.0};
static int3 AC_INTERNAL_global_int_vec = {0,0,0};

static AcReal AC_INTERNAL_big_real_array[8*1024*1024]{0.0};
static int AC_INTERNAL_big_int_array[8*1024*1024]{0};

//the int key in the nested map corresponds to the starting vertexIdx linearized
std::unordered_map<Kernel,std::unordered_map<int,int>> reduce_offsets;
int kernel_running_reduce_offsets[NUM_KERNELS];

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
get_bpg(const Volume dims, const Volume tpb)
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

bool
is_valid_configuration(const Volume dims, const Volume tpb)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  const size_t warp_size = props.warpSize;
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
    if (tpb.z > 1)
      return false;
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

__device__ __constant__ AcMeshInfo d_mesh_info;
#include "dconst_arrays_decl.h"
//TP: We do this ugly macro because I want to keep the generated headers the same if we are compiling cpu analysis and for the actual gpu comp
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME) __device__ __constant__ DATATYPE* gmem_##DEFINE_NAME##_arrays[NUM_##ARR_NAME##_ARRAYS+1] 
#include "gmem_arrays_decl.h"



//The macros above generate d arrays like these:

// Astaroth 2.0 backwards compatibility START
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])


#include "dconst_decl.h"


#include "get_address.h"

#include "load_and_store_array.h"



#define DEVICE_VTXBUF_IDX(i, j, k)                                             \
  ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))

__device__ int
LOCAL_COMPDOMAIN_IDX(const int3 coord)
{
  return (coord.x) + (coord.y) * DCONST(AC_nx) + (coord.z) * DCONST(AC_nxy);
}

__device__ constexpr int
IDX(const int i)
{
  return i;
}

#if 1
__device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
  return DEVICE_VTXBUF_IDX(i, j, k);
}
#else
constexpr __device__ int
IDX(const uint i, const uint j, const uint k)
{
  /*
  const int precision   = 32; // Bits
  const int dimensions  = 3;
  const int bits = ceil(precision / dimensions);
  */
  const int dimensions = 3;
  const int bits       = 11;

  uint idx = 0;
#pragma unroll
  for (uint bit = 0; bit < bits; ++bit) {
    const uint mask = 0b1 << bit;
    idx |= ((i & mask) << 0) << (dimensions - 1) * bit;
    idx |= ((j & mask) << 1) << (dimensions - 1) * bit;
    idx |= ((k & mask) << 2) << (dimensions - 1) * bit;
  }
  return idx;
}
#endif

// Only used in reductions
__device__ __forceinline__ int
IDX(const int3 idx)
{
  return DEVICE_VTXBUF_IDX(idx.x, idx.y, idx.z);
}

//#define Field3(x, y, z) make_int3((x), (y), (z))
#define print printf                          // TODO is this a good idea?
#define len(arr) sizeof(arr) / sizeof(arr[0]) // Leads to bugs if the user
// passes an array into a device function and then calls len (need to modify
// the compiler to always pass arrays to functions as references before
// re-enabling)

#include "random.cuh"

#include "user_dfuncs.h"
#define suppress_unused_warning(X) (void)X
#include "user_kernels.h"
#include "extern_kernels.h"


typedef struct {
  Kernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;


static TBConfig getOptimalTBConfig(const Kernel kernel, const int3 dims, VertexBufferArray vba);

static __global__ void
flush_kernel(AcReal* arr, const size_t n, const AcReal value)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    arr[idx] = value;
}

AcResult
acKernelFlush(const cudaStream_t stream, AcReal* arr, const size_t n,
              const AcReal value)
{
  const size_t tpb = 256;
  const size_t bpg = (size_t)(ceil((double)n / tpb));
  flush_kernel<<<bpg, tpb, 0, stream>>>(arr, n, value);
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
acVBAReset(const cudaStream_t stream, VertexBufferArray* vba)
{
  const size_t count = vba->bytes / sizeof(vba->in[0][0]);

  // Set vba.in data to all-nan and vba.out to 0
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    if (vtxbuf_is_auxiliary[i])
    {
      acKernelFlush(stream, vba->in[i],count, (AcReal)0.0);
    } else{
      acKernelFlush(stream, vba->in[i],count, (AcReal)NAN);
      acKernelFlush(stream, vba->out[i],count, (AcReal)0.0);
    }
  }
  memset(&vba->kernel_input_params,0,sizeof(acKernelInputParams));
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
template <typename P>
void
memcpy_to_gmem_array(const P param, void* &ptr)
{
#include "memcpy_to_gmem_arrays.h"
}
template <typename P>
void
memcpy_from_gmem_array(const P param, void* &ptr)
{
#include "memcpy_from_gmem_arrays.h"
}


template <typename P>
struct allocate_arrays
{
	void operator()(const AcMeshInfo& config) 
	{
		for(P array : get_params<P>())
		{
			if(get_config_param(array,config) != nullptr && !is_dconst(array))
			{
				void* d_mem_ptr;
			        device_malloc(&d_mem_ptr, sizeof(get_config_param(array,config)[0])*get_array_length(array,config));
				memcpy_to_gmem_array(array,d_mem_ptr);
			}
		}
	}
};


VertexBufferArray
acVBACreate(const AcMeshInfo config)
{
  //can't use acVertexBufferDims because of linking issues
#if TWO_D == 0
  const int3 counts = (int3){
        (config.int_params[AC_mx]),
        (config.int_params[AC_my]),
        (config.int_params[AC_mz])
  };
#else
  const int3 counts = (int3){
        (config.int_params[AC_mx]),
        (config.int_params[AC_my]),
	1,
  };
#endif

  VertexBufferArray vba;
  size_t count = counts.x*counts.y*counts.z;
  size_t bytes = sizeof(vba.in[0][0]) * count;
  vba.bytes          = bytes;
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
    //Allocate auxilary fields
    //They need only a single copy so out can point to in
    if (vtxbuf_is_auxiliary[i])
    {
      device_malloc((void**) &vba.in[i],bytes);
      vba.out[i] = vba.in[i];
    }else{
      device_malloc((void**) &vba.in[i],bytes);
      device_malloc((void**) &vba.out[i],bytes);
    }
  }
#endif
  //Allocate workbuffers
  for (int i = 0; i < NUM_WORK_BUFFERS; ++i)
    device_malloc((void**)&vba.w[i],bytes);


  AcArrayTypes::run<allocate_arrays>(config);

  acVBAReset(0, &vba);
  cudaDeviceSynchronize();
  return vba;
}

template <typename P>
struct update_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array : get_params<P>())
		{
			if(is_dconst(array)) continue;
			auto config_array = get_config_param(array,config);
			void* gmem_array;
			memcpy_from_gmem_array(array,gmem_array);
			size_t bytes = sizeof(config_array[0])*get_array_length(array,config);
			if(config_array == nullptr && gmem_array != nullptr) 
				device_free(&gmem_array,bytes);
			else if(config_array != nullptr && gmem_array  == nullptr) 
				device_malloc(&gmem_array,bytes);
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};
void
acVBAUpdate(VertexBufferArray* vba, const AcMeshInfo config)
{
  (void)vba;
  AcArrayTypes::run<update_arrays>(config);
}

template <typename P>
struct free_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array: get_params<P>())
		{
			auto config_array = get_config_param(array,config);
			void* gmem_array;
			memcpy_from_gmem_array(array,gmem_array);
			if(config_array == nullptr ||is_dconst(array)) continue;
			device_free(&gmem_array, get_array_length(array,config));
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};
void
acVBADestroy(VertexBufferArray* vba, const AcMeshInfo config)
{
  for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) { device_free(&(vba->in[i]), vba->bytes);
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
  vba->bytes = 0;
}
int
get_num_of_reduce_output(const dim3 bpg, const dim3 tpb)
{
#if AC_USE_HIP
	const int warp_size = rocprim::host_warp_size();
#else
	const int warp_size = 32;
#endif
	const int num_of_warps_per_block = (tpb.x*tpb.y*tpb.z + warp_size-1)/warp_size;
	const int num_of_blocks = bpg.x*bpg.y*bpg.z;
	return num_of_warps_per_block*num_of_blocks;
}

int
get_kernel_index(const Kernel kernel)
{
	for(int i = 0; i < NUM_KERNELS; ++i)
		if(kernel == kernels[i]) return i;
	return -1;
}
AcResult
acLaunchKernel(Kernel kernel, const cudaStream_t stream, const int3 start,
               const int3 end, VertexBufferArray vba)
{
  const int3 n = end - start;

  const TBConfig tbconf = getOptimalTBConfig(kernel, n, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));

  const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));
  const int key = start.x + 10000*start.y + 10000*10000*start.z;
  if(reduce_offsets[kernel].find(key) == reduce_offsets[kernel].end())
  {
  	reduce_offsets[kernel][key] = kernel_running_reduce_offsets[get_kernel_index(kernel)];
  	kernel_running_reduce_offsets[get_kernel_index(kernel)] += get_num_of_reduce_output(bpg,tpb);
  }

  vba.reduce_offset = reduce_offsets[kernel][key];
  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
  kernel<<<bpg, tpb, smem, stream>>>(start, end, vba);
  ERRCHK_CUDA_KERNEL();

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}

AcResult
acBenchmarkKernel(Kernel kernel, const int3 start, const int3 end,
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
  kernel<<<bpg, tpb, smem>>>(start, end, vba);
  cudaEventRecord(tstop);
  cudaEventSynchronize(tstop);
  ERRCHK_CUDA_KERNEL();
  cudaDeviceSynchronize();

  // Benchmark
  cudaEventRecord(tstart); // Timing start
  kernel<<<bpg, tpb, smem>>>(start, end, vba);
  cudaEventRecord(tstop); // Timing stop
  cudaEventSynchronize(tstop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, tstart, tstop);

  size_t kernel_id = NUM_KERNELS;
  for (size_t i = 0; i < NUM_KERNELS; ++i) {
    if (kernels[i] == kernel) {
      kernel_id = i;
    }
  }
  ERRCHK_ALWAYS(kernel_id < NUM_KERNELS);
  printf("Kernel %s time elapsed: %g ms\n", kernel_names[kernel_id],
         (double)milliseconds);

  // Timer destroy
  cudaEventDestroy(tstart);
  cudaEventDestroy(tstop);

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}


#if TWO_D == 0
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
#else
AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t /* stream */,
              const AcReal data[STENCIL_HEIGHT][STENCIL_WIDTH])
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

  const size_t bytes = sizeof(data[0][0]) * STENCIL_HEIGHT * STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyToSymbol(
      stencils, data, bytes, stencil * bytes, cudaMemcpyHostToDevice);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};
#endif

#if TWO_D == 0
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
#else
AcResult
acStoreStencil(const Stencil stencil, const cudaStream_t /* stream */,
               AcReal data[STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  // Ensure all acLoadUniform calls have completed before continuing
  cudaDeviceSynchronize();

  const size_t bytes = sizeof(data[0][0]) * STENCIL_HEIGHT * STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyFromSymbol(
      data, stencils, bytes, stencil * bytes, cudaMemcpyDeviceToHost);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};
#endif



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
	cudaDeviceSynchronize();
	ERRCHK_ALWAYS(values  != nullptr);
	const size_t bytes = length*sizeof(values[0]);
	if(!is_dconst(array))
	{
		void* dst_ptr;
		memcpy_from_gmem_array(array,dst_ptr);
		ERRCHK_ALWAYS(dst_ptr != nullptr);
		ERRCHK_CUDA_ALWAYS(cudaMemcpy(dst_ptr,values,bytes,cudaMemcpyHostToDevice));
	}
	else
	{
		const size_t offset = (size_t) get_dconst_array_offset(array)*sizeof(V);
		ERRCHK_CUDA_ALWAYS(load_array(values, bytes, offset));
	}
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
	if(!is_dconst(array))
	{
		void* src_ptr;
		memcpy_from_gmem_array(array,src_ptr);
		ERRCHK_ALWAYS(src_ptr != nullptr);
		ERRCHK_CUDA_ALWAYS(cudaMemcpyFromSymbol(values, src_ptr, bytes, 0, cudaMemcpyDeviceToHost));
	}
	else
	{
		const size_t offset = (size_t) get_dconst_array_offset(array)*sizeof(V);
		ERRCHK_CUDA_ALWAYS(store_array(values, bytes, offset));
	}
	return AC_SUCCESS;
}


#include "load_and_store_uniform_funcs.h"


static TBConfig
autotune(const Kernel kernel, const int3 dims, VertexBufferArray vba)
{
  vba.reduce_offset = 0;
  ERRCHK_ALWAYS(get_kernel_index(kernel) < NUM_KERNELS);
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

#if TWO_D == 0
  const int3 start = (int3){
      STENCIL_ORDER / 2,
      STENCIL_ORDER / 2,
      STENCIL_ORDER / 2,
  };
#else
  const int3 start = (int3){
      STENCIL_ORDER / 2,
      STENCIL_ORDER / 2,
      0,
  };
#endif
  const int3 end = start + dims;

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
  for (int z = 1; z <= max_threads_per_block; ++z) {
    for (int y = 1; y <= max_threads_per_block; ++y) {
      // 64 bytes on NVIDIA but the minimum L1 cache transaction is 32
      const int minimum_transaction_size_in_elems = 32 / sizeof(AcReal);
      for (int x = minimum_transaction_size_in_elems;
           x <= max_threads_per_block; x += minimum_transaction_size_in_elems) {

        if (x * y * z > max_threads_per_block)
          break;

        // if (x * y * z * max_regs_per_thread > max_regs_per_block)
        //  break;

        // if (max_regs_per_block / (x * y * z) < min_regs_per_thread)
        //   continue;

        // if (x < y || x < z)
        //   continue;

        const dim3 tpb(x, y, z);
        const dim3 bpg    = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));
        const size_t smem = get_smem(to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        if (smem > max_smem)
          continue;

        if ((x * y * z) % props.warpSize)
          continue;

        if (!is_valid_configuration(to_volume(dims), to_volume(tpb)))
          continue;

        // #if VECTORIZED_LOADS
        //         const size_t window = tpb.x + STENCIL_ORDER;

        //         // Vectorization criterion
        //         if (window % veclen) // Window not divisible into vectorized
        //         blocks
        //           continue;

        //         if (dims.x % tpb.x)
        //           continue;

        //           // May be too strict
        //           // if (dims.x % tpb.x || dims.y % tpb.y || dims.z % tpb.z)
        //           //   continue;
        // #endif
        // #if 0 // Disabled for now (waiting for cleanup)
        // #if USE_SMEM
        //         const size_t max_smem = 128 * 1024;
        //         if (smem > max_smem)
        //           continue;

        // #if VECTORIZED_LOADS
        //         const size_t window = tpb.x + STENCIL_ORDER;

        //         // Vectorization criterion
        //         if (window % veclen) // Window not divisible into vectorized
        //         blocks
        //           continue;

        //         if (dims.x % tpb.x || dims.y % tpb.y || dims.z % tpb.z)
        //           continue;
        // #endif

        //           //  Padding criterion
        //           //  TODO (cannot be checked here)
        // #else
        //         if ((x * y * z) % warp_size)
        //           continue;
        // #endif
        // #endif

        // printf("%d, %d, %d: %lu\n", tpb.x, tpb.y, tpb.z, smem);

        cudaEvent_t tstart, tstop;
        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        kernel<<<bpg, tpb, smem>>>(start, end, vba); // Dryrun
        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < num_iters; ++i)
          kernel<<<bpg, tpb, smem>>>(start, end, vba);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        cudaEventDestroy(tstart);
        cudaEventDestroy(tstop);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        if (cudaGetLastError() != cudaSuccess) {
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
    }
  }
  c.tpb = best_tpb;

  // printf("\tThe best tpb: (%d, %d, %d), time %f ms\n", best_tpb.x,
  // best_tpb.y,
  //        best_tpb.z, (double)best_time / num_iters);

  FILE* fp = fopen(autotune_csv_path, "a");
  ERRCHK_ALWAYS(fp);
#if IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
  fprintf(fp, "%d, (%d, %d, %d), (%d, %d, %d), %g\n", IMPLEMENTATION, nx, ny,
          nz, best_tpb.x, best_tpb.y, best_tpb.z,
          (double)best_time / num_iters);
#else
  fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %d, %g\n", IMPLEMENTATION, get_kernel_index(kernel), dims.x,
          dims.y, dims.z, best_tpb.x, best_tpb.y, best_tpb.z,
          (double)best_time / num_iters);
#endif
  fclose(fp);

  if (c.tpb.x * c.tpb.y * c.tpb.z <= 0) {
    fprintf(stderr,
            "Fatal error: failed to find valid thread block dimensions.\n");
  }
  ERRCHK_ALWAYS(c.tpb.x * c.tpb.y * c.tpb.z > 0);
  return c;
}
int
get_entries(char** dst, const char* line)
{
      char* line_copy = strdup(line);
      int counter = 0;
      char* token;
      token = strtok(line_copy,",");
      while(token != NULL)
      {
              dst[counter] = strdup(token);
              ++counter;
              token = strtok(NULL,",");
      }
      free(line_copy);
      return counter;
}
static int3
read_optim_tpb(const Kernel kernel, const int3 dims)
{
  const char* filename = autotune_csv_path;
  FILE *file = fopen ( filename, "r" );
  int3 res = {-1,-1,-1};
  const double best_time = pow(10.0,20);
  if (file != NULL) {
    char line [1000];
    while(fgets(line,sizeof line,file)!= NULL) /* read a line from a file */ {
      char* entries[9];
      int num_entries = get_entries(entries,line);
      if(num_entries > 1)
      {
         int kernel_index  = atoi(entries[1]);
         int3 read_dims = {atoi(entries[2]), atoi(entries[3]), atoi(entries[4])};
         int3 tpb = {atoi(entries[5]), atoi(entries[6]), atoi(entries[7])};
         double time = atof(entries[8]);
	 res =  (read_dims == dims && kernel_index == get_kernel_index(kernel) && time < best_time) ? tpb  : res;
      }
      for(int i = 0; i < num_entries; ++i)
             free(entries[i]);

  }
    fclose(file);
  }
  else {
    perror(filename); //print the error message on stderr.
  }
  return res;
}


static TBConfig
getOptimalTBConfig(const Kernel kernel, const int3 dims, VertexBufferArray vba)
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

Kernel
GetOptimizedKernel(const AcKernel kernel_enum, const VertexBufferArray vba)
{
	#include "user_kernel_ifs.h"
	//silence unused warnings
	(void)vba;
	return kernels[(int) kernel_enum];
}

