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
#define rocprim__warpSize() rocprim::warp_size()
#define rocprim__warpId()   rocprim::warp_id()
#define rocprim__warp_shuffle_down rocprim::warp_shuffle_down
#define rocprim__warp_shuffle rocprim::warp_shuffle

#include "acc_runtime.h"
#include "ac_buffer.h"

#include "user_defines_runtime_lib.h"

#include "../acc/string_vec.h"
typedef void (*Kernel)(const int3, const int3, DeviceVertexBufferArray vba);
#define AcReal3(x,y,z)   (AcReal3){x,y,z}
#define AcComplex(x,y)   (AcComplex){x,y}
static AcBool3 dimension_inactive{};
static int3 raytracing_subblock{};
static int  x_ray_shared_mem_block_size{};
static int  z_ray_shared_mem_block_size{};
static bool sparse_autotuning=false;
static int3    max_tpb_for_reduce_kernels{100,100,100};
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
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
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
typedef struct
{
	void* data;
	size_t bytes;
} AcDeviceTmpBuffer;

static dim3 last_tpb = (dim3){0, 0, 0};
struct Int3Hash {
    std::size_t operator()(const int3& v) const {
        return std::hash<int>()(v.x) ^ std::hash<int>()(v.y) << 1 ^ std::hash<int>()(v.z) << 2;
    }
};
std::array<std::unordered_map<int3,int,Int3Hash>,NUM_KERNELS> reduce_offsets;
int kernel_running_reduce_offsets[NUM_KERNELS];

AcAutotuneMeasurement
return_own_measurement(const AcAutotuneMeasurement local_measurement) {return local_measurement;}

static int grid_pid = 0;
[[maybe_unused]] static int nprocs   = 0;
static AcMeasurementGatherFunc gather_func  = return_own_measurement;
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
#include "../../src/helpers/ceil_div.h"

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
        as_size_t(ceil_div(dims.x,tpb.x)),
        as_size_t(ceil_div(dims.y,tpb.y)),
        as_size_t(ceil_div(dims.z,tpb.z)),
    };
  }
  default: {
    ERROR("Invalid IMPLEMENTATION in get_bpg");
    return (Volume){0, 0, 0};
  }
  }
}
#include "stencil_accesses.h"
#include "../acc/mem_access_helper_funcs.h"

static bool
is_raytracing_kernel(const AcKernel kernel)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray]) return true;
	}
	return false;
}

static int
num_fields_ray_accessed_read_and_written(const AcKernel kernel)
{
	int res = 0;
	for(int field = 0; field < NUM_ALL_FIELDS; ++field)
	{
		if(write_called[kernel][field] || stencils_accessed[kernel][field][0])
		{
			res++;
			continue;
		}
		for(int ray = 0; ray < NUM_RAYS; ++ray)
		{
			if(
			    incoming_ray_value_accessed[kernel][field][ray]
			    || outgoing_ray_value_accessed[kernel][field][ray]
			    )
			{
				res++;
				continue;
			}
		}
	}
	return res;
}

static AcBool3
raytracing_step_direction(const AcKernel kernel)
{
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray])
			{
				if(ray_directions[ray].z != 0) return (AcBool3){false,false,true};
				if(ray_directions[ray].y != 0) return (AcBool3){false,true,false};
				if(ray_directions[ray].x != 0) return (AcBool3){true,false,false};
			}
	}
	return (AcBool3){false,false,false};
}
static AcBool3
raytracing_directions(const AcKernel kernel)
{
	AcBool3 res = (AcBool3){false,false,false};
	for(int ray = 0; ray < NUM_RAYS; ++ray)
	{
		for(int field = 0; field < NUM_ALL_FIELDS; ++field)
			if(incoming_ray_value_accessed[kernel][field][ray])
			{
				res.x |= ray_directions[ray].x != 0;
				res.y |= ray_directions[ray].y != 0;
				res.z |= ray_directions[ray].z != 0;
			}
	}
	return res;
}
static int
raytracing_number_of_directions(const AcKernel kernel)
{
	const auto dirs = raytracing_directions(kernel);
	return dirs.x+dirs.y+dirs.z;
}

static bool
is_coop_raytracing_kernel(const AcKernel kernel)
{
	return is_raytracing_kernel(kernel) && (raytracing_number_of_directions(kernel) > 1);
}

Volume
get_bpg(Volume dims, const AcKernel kernel, const int3 block_factors, const Volume tpb)
{
	if(kernel_has_block_loops(kernel)) return get_bpg(ceil_div(dims,block_factors), tpb);
	return get_bpg(dims,tpb);
}

static cudaDeviceProp
get_device_prop()
{
  cudaDeviceProp props;
  ERRCHK_CUDA_ALWAYS(cudaGetDeviceProperties(&props, 0));
  return props;
}


static int3
get_ghosts()
{
  return (int3){
	  dimension_inactive.x ? 0 : NGHOST,
	  dimension_inactive.y ? 0 : NGHOST,
	  dimension_inactive.z ? 0 : NGHOST
  };
}
template <typename T>
bool
is_large_launch(const T dims)
{
  const int3 ghosts = get_ghosts();
  return ((int)dims.x > ghosts.x && (int)dims.y > ghosts.y && (int)dims.z > ghosts.z);
}


bool
is_valid_configuration(const Volume dims, const Volume tpb, const AcKernel kernel)
{
  const size_t warp_size    = get_device_prop().warpSize;
  const size_t xmax         = (size_t)(warp_size * ceil_div(dims.x,warp_size));
  const size_t ymax         = (size_t)(warp_size * ceil_div(dims.y,warp_size));
  const size_t zmax         = (size_t)(warp_size * ceil_div(dims.z,warp_size));
  const bool too_large      = (tpb.x > xmax) || (tpb.y > ymax) || (tpb.z > zmax);
  const bool not_full_warp  = (tpb.x*tpb.y*tpb.z < warp_size);
  if(is_coop_raytracing_kernel(kernel))
  {
	int maxBlocksPerSM{};
	ERRCHK_CUDA_ALWAYS(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&maxBlocksPerSM,
			kernels[kernel],
			tpb.x*tpb.y*tpb.z,
			0
	));
  	const auto bpg = get_bpg(dims,to_volume(tpb));
	const int totalMaxBlocks = get_device_prop().multiProcessorCount*maxBlocksPerSM;
	if((int)(bpg.x*bpg.y*bpg.z) > totalMaxBlocks) return false;
  }
  if(raytracing_step_direction(kernel).x)
  {
	if(((int)tpb.y - (x_ray_shared_mem_block_size)) != 0) return false;
  }

  else
  {
  	//TP: in most cases this a reasonable limitation but at least theoretically the shape of the threadblock might be more important
  	//    than warp considerations. So impose this limitation only if the user allows it
  	if (sparse_autotuning && (dims.x >= warp_size && tpb.x % warp_size != 0)) return false;
  }

  if(kernel_reduces_profile(kernel))
  {
	  if(tpb.y > (size_t)max_tpb_for_reduce_kernels.y) return false;
	  if(tpb.z > (size_t)max_tpb_for_reduce_kernels.z) return false;
	  //TP: if we enforce that tpb.x is a multiple of the warp size then 
	  //can easily do warp reduce while doing a reduction whose result is not x-dependent --> major saving in memory and performance increase
	  if(dims.x >= warp_size && tpb.x % warp_size != 0) return false;
  }
//  const bool single_tb      = (tpb.x >= dims.x) && (tpb.y >= dims.y) && (tpb.z >= dims.z);

  //TP: if not utilizing the whole warp invalid, expect if dims are so small that could not utilize a whole warp 
  if(not_full_warp && is_large_launch(dims)) return false;

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

const bool SHARED_MEM_Z_RAYS = false;
size_t
get_smem(const AcKernel kernel, const Volume tpb, const size_t stencil_order,
         const size_t bytes_per_elem)
{
  if(is_raytracing_kernel(kernel) && raytracing_step_direction(kernel).x)
  {
	//TP: we pad the y dimension by one to avoid bank conflicts
	return bytes_per_elem*(tpb.y+1)*tpb.z*(x_ray_shared_mem_block_size+2)*num_fields_ray_accessed_read_and_written(kernel);
  }
  if(is_raytracing_kernel(kernel) && raytracing_step_direction(kernel).z && SHARED_MEM_Z_RAYS)
  {
	return bytes_per_elem*(tpb.x+2)*(tpb.y+2)*(z_ray_shared_mem_block_size+2)*num_fields_ray_accessed_read_and_written(kernel);
  }
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
#define DECLARE_CONST_DIMS_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, LEN) static __device__ DATATYPE AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME[LEN]
#include "gmem_arrays_decl.h"

AcReal
get_reduce_state_flush_var_real(const AcReduceOp state)
{
	return 
		(state == NO_REDUCE || state == REDUCE_SUM) ? 0.0 :
		(state == REDUCE_MIN) ? AC_REAL_MAX :
		(state == REDUCE_MAX) ? -AC_REAL_MAX :
		0.0;
}

int
get_reduce_state_flush_var_int(const AcReduceOp state)
{
	return 
		(state == NO_REDUCE || state == REDUCE_SUM) ? 0 :
		(state == REDUCE_MIN) ? INT_MAX:
		(state == REDUCE_MAX) ? -INT_MAX:
		0;
}

#if AC_DOUBLE_PRECISION
float
get_reduce_state_flush_var_float(const AcReduceOp state)
{
	return 
		(state == NO_REDUCE || state == REDUCE_SUM) ? 0.0 :
		(state == REDUCE_MIN) ? FLT_MAX :
		(state == REDUCE_MAX) ? -FLT_MAX :
		0.0;
}
#endif

typedef struct {
  AcKernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;


static TBConfig getOptimalTBConfig(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba);


template <typename T>
T TO_CORRECT_ORDER(const T vol)
{
	return vol;
}
size_t TO_CORRECT_ORDER(const size_t size)
{
	return size;
}


#define KERNEL_LAUNCH(func,bgp,tpb,...) \
	func<<<TO_CORRECT_ORDER(bpg),TO_CORRECT_ORDER(tpb),__VA_ARGS__>>>

#define KERNEL_VBA_LAUNCH(func,bgp,tpb,...) \
	func<<<TO_CORRECT_ORDER(bpg),TO_CORRECT_ORDER(tpb),__VA_ARGS__>>>


__device__ __constant__ AcReal* d_symbol_reduce_scratchpads_real[NUM_REAL_SCRATCHPADS];
static AcReal* d_reduce_scratchpads_real[NUM_REAL_SCRATCHPADS];
static size_t d_reduce_scratchpads_size_real[NUM_REAL_SCRATCHPADS];
__device__ __constant__ AcReal  d_reduce_real_res_symbol[NUM_REAL_SCRATCHPADS];

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




#include "reduce_helpers.h"


void
resize_scratchpads_to_fit(const size_t n_elems, VertexBufferArray vba, const AcKernel kernel)
{
	resize_reals_to_fit(n_elems,vba,kernel);
	resize_ints_to_fit(n_elems,vba,kernel);
#if AC_DOUBLE_PRECISION
	resize_floats_to_fit(n_elems,vba,kernel);
#endif
}

size_t
acGetRealScratchpadSize(const size_t i)
{
	return d_reduce_scratchpads_size_real[i];
}

//The macros above generate d arrays like these:

// Astaroth 2.0 backwards compatibility START
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])


#define DEVICE_INLINE __device__ __forceinline__
#include "dconst_decl.h"
#include "output_value_decl.h"



#include "get_address.h"
#include "load_dconst_arrays.h"
#include "store_dconst_arrays.h"

#define PROFILE_X_Y_OR_Z_INDEX(i,j) \
  ((i) + (j)*VAL(AC_mlocal).x)

#define PROFILE_Y_X_OR_Z_INDEX(i,j) \
  ((i) + (j)*VAL(AC_mlocal).y)

#define PROFILE_Z_X_OR_Y_INDEX(i,j) \
  ((i) + (j)*VAL(AC_mlocal).z)


#define DEVICE_VTXBUF_IDX(i, j, k)                                             \
  ((i) + (j)*VAL(AC_mlocal).x + (k)*VAL(AC_mlocal_products).xy)

#define DEVICE_VARIABLE_VTXBUF_IDX(i, j, k,dims)                                             \
  ((i) + dims.x*((j) + (k)*dims.y))

#define LOCAL_COMPDOMAIN_IDX(coord) \
	((coord.x) + (coord.y) * VAL(AC_nlocal).x + (coord.z) * VAL(AC_nlocal_products).xy)

#define print printf                          // TODO is this a good idea?
// passes an array into a device function and then calls len (need to modify
// the compiler to always pass arrays to functions as references before
// re-enabling)

#include "random.cuh"

#define suppress_unused_warning(X) (void)X
#define longlong long long
#define size(arr) (int)(sizeof(arr) / sizeof(arr[0])) // Leads to bugs if the user
#define error_message(error,message) 
#define fatal_error_message(error,message) 

__device__
AcReal
safe_access(const AcReal* arr, const int dims, const int index, const char* name)
{
	if(arr == NULL)
	{
		printf("Trying to access %s which is NULL!\n",name);
		//TP: assert is not defined on Mahti :(
		//assert(false);
		return 0.0;
	}
	else if(index < 0 || index >= dims)
	{
		printf("Trying to access %s out of bounds!: %d\n",name,index);
		//TP: assert is not defined on Mahti :(
		//assert(false);
		return 0.0;
	}
	return arr[index];
}
__device__ UNUSED
AcReal
safe_access(const AcReal* arr, const int dims, const int index, const AcRealArrayParam param)
{
	return safe_access(arr,dims,index,real_array_names__device__[param]);
}

#include "device_fields_info.h"

static __device__ UNUSED
int3
ac_get_field_halos(const Field& field)
{
	return VAL(vtxbuf_device_halos[field]);
}


#define postprocess_reduce_result(DST,OP)
#include "user_kernels.h"
#undef size
#undef longlong


template<typename T1, typename T2>
AcResult
acLaunchKernelVariadic1d(AcKernel kernel, const cudaStream_t stream, const size_t start, const size_t end,T1 param1, T2 param2)
{
  const Volume volume_start = {start,0,0};
  const Volume volume_end   = {end,1,1};
  VertexBufferArray vba{};
  acLoadKernelParams(vba.on_device.kernel_input_params,kernel,param1,param2); 
  return acLaunchKernel(kernel,stream,volume_start,volume_end,vba);
}

template<typename T1, typename T2>
AcResult
acLaunchKernelVariadic1d(AcKernel kernel, const int stream, const int start, const size_t end,T1 param1, T2 param2)
{
  const Volume volume_start = {as_size_t(start),0,0};
  const Volume volume_end   = {end,1,1};
  VertexBufferArray vba{};
  acLoadKernelParams(vba.on_device.kernel_input_params,kernel,param1,param2); 
  return acLaunchKernel(kernel,cudaStream_t(stream),volume_start,volume_end,vba);
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


#include "user_built-in_constants.h"
#include "user_builtin_non_scalar_constants.h"



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
acPBAReset(const cudaStream_t stream, ProfileBufferArray* pba, const AcMeshDims* dims)
{
  // Set pba.in data to all-nan and pba.out to 0
  for (int i = 0; i < NUM_PROFILES; ++i) {
    acKernelFlush(stream, pba->in[i],  prof_count(Profile(i),dims[i].m1), (AcReal)NAN);
    acKernelFlush(stream, pba->out[i], prof_count(Profile(i),dims[i].m1), (AcReal)0);
  }
  return AC_SUCCESS;
}
size_t
get_amount_of_device_memory_free()
{
	size_t free_mem, total_mem;
	ERRCHK_CUDA_ALWAYS(acMemGetInfo(&free_mem,&total_mem));
	return free_mem;
}
void
device_malloc(void** dst, const size_t bytes)
{
  if(get_amount_of_device_memory_free() < bytes)
  {
	fprintf(stderr,"Tried to allocate %ld bytes but have only %ld bytes of memory left on the device\n", bytes, get_amount_of_device_memory_free());
  	ERRCHK_ALWAYS(get_amount_of_device_memory_free() >= bytes);
  }
 #if USE_COMPRESSIBLE_MEMORY 
    ERRCHK_CUDA_ALWAYS(mallocCompressible(dst, bytes));
 #else
    ERRCHK_CUDA_ALWAYS(acMalloc(dst, bytes));
  #endif
  ERRCHK_ALWAYS(dst != NULL);
}
void
device_malloc(AcReal** dst, const size_t bytes)
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
  ERRCHK_CUDA_ALWAYS(acFree(*dst));
  //used to silence unused warning
  (void)bytes;
#endif
  *dst = NULL;
}

size_t
device_resize(void** dst,const size_t old_bytes,const size_t new_bytes)
{
	if(old_bytes >= new_bytes) return old_bytes;
	if(old_bytes) device_free(dst,old_bytes);
	device_malloc(dst,new_bytes);
	return new_bytes;
}


ProfileBufferArray
acPBACreate(const AcMeshDims* dims)
{
  ProfileBufferArray pba{};
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),dims[i].m1)*sizeof(AcReal);
    device_malloc(&pba.in[i],  bytes);
    device_malloc(&pba.out[i], bytes);
    //pba.out[i] = pba.in[i];
  }

  acPBAReset(0, &pba, dims);
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
  return pba;
}

void
acPBADestroy(ProfileBufferArray* pba, const AcMeshDims* dims)
{
  for (int i = 0; i < NUM_PROFILES; ++i) {
    const size_t bytes = prof_size(Profile(i),dims[i].m1)*sizeof(AcReal);
    device_free(&pba->in[i],  bytes);
    device_free(&pba->out[i], bytes);
    pba->in[i]  = NULL;
    pba->out[i] = NULL;
  }
}

AcResult
acVBAReset(const cudaStream_t stream, VertexBufferArray* vba)
{

  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    ERRCHK_ALWAYS(vba->on_device.in[i]);
    ERRCHK_ALWAYS(vba->on_device.out[i]);
    acKernelFlush(stream, vba->on_device.in[i], vba->counts[i], (AcReal)NAN);
    acKernelFlush(stream, vba->on_device.out[i], vba->counts[i], (AcReal)0.0);
  }

  const AcComplex zero_complex{0.0,0.0};
  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
    size_t n = vba->computational_dims.m1.x*vba->computational_dims.m1.y*vba->computational_dims.m1.z;
    ERRCHK_ALWAYS(vba->on_device.complex_in[field]);
    acKernelFlush(stream, vba->on_device.complex_in[field],n,zero_complex);
  }
  memset(&vba->on_device.kernel_input_params,0,sizeof(acKernelInputParams));
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBAReset(stream, &vba->on_device.profiles, vba->profile_dims);
  return AC_SUCCESS;
}


template <typename T>
void
device_malloc(T** dst, const int bytes)
{
 #if USE_COMPRESSIBLE_MEMORY 
    ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)dst, bytes));
 #else
    ERRCHK_CUDA_ALWAYS(acMalloc((void**)dst, bytes));
  #endif
}

#include "memcpy_to_gmem_arrays.h"

#include "memcpy_from_gmem_arrays.h"

template <typename P>
struct allocate_arrays
{
	void operator()(const AcMeshInfo& config) 
	{
		for(P array : get_params<P>())
		{
			if(config[array] == nullptr && is_accessed(array))
			{
				fprintf(stderr,"Passed %s as NULL but it is accessed kernels!!\n",get_name(array));
				fflush(stderr);
				ERRCHK_ALWAYS(config[array] != nullptr);
			}
			if (config[array] != nullptr && !is_dconst(array) && is_alive(array))
			{

#if AC_VERBOSE
				fprintf(stderr,"Allocating %s|%zu\n",get_name(array),get_array_length(array,config));
				fflush(stderr);
#endif
				auto d_mem_ptr = get_empty_pointer(array);
			        device_malloc(((void**)&d_mem_ptr), sizeof(config[array][0])*get_array_length(array,config));
				memcpy_to_gmem_array(array,d_mem_ptr);
			}
		}
	}
};


#if AC_USE_HIP
#include <hipcub/hipcub.hpp>
#define cub hipcub
#else
#include <cub/cub.cuh>
#endif

template <typename T>
void
cub_reduce(AcDeviceTmpBuffer& temp_storage, const cudaStream_t stream, const T* d_in, const size_t count, T* d_out,  AcReduceOp reduce_op)
{
  switch(reduce_op)
  {
	  case(REDUCE_SUM):
	  	ERRCHK_CUDA(cub::DeviceReduce::Sum(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	  case(REDUCE_MIN):
	  	ERRCHK_CUDA(cub::DeviceReduce::Min(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	  case(REDUCE_MAX):
	  	ERRCHK_CUDA(cub::DeviceReduce::Max(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	default:
		ERRCHK_ALWAYS(reduce_op != NO_REDUCE);
  }
  if (acGetLastError() != cudaSuccess) {
          ERRCHK_CUDA_KERNEL_ALWAYS();
          ERRCHK_CUDA_ALWAYS(acGetLastError());
  }
}

size3_t
acGetProfileReduceScratchPadDims(const int profile, const AcMeshDims dims)
{
	const auto type = prof_types[profile];
    	if(type == PROFILE_YZ || type == PROFILE_ZY)
    		return
    		{
    		    	    dims.reduction_tile.x,
    		    	    dims.m1.y,
    		    	    dims.m1.z
    		};
    	if(type == PROFILE_XZ || type == PROFILE_ZX)
		return
    		{
    		    	    dims.m1.x,
    		    	    dims.reduction_tile.y,
    		    	    dims.m1.z
    		};
    	if(type == PROFILE_YX || type == PROFILE_XY)
		return
    		{
    		    	    dims.m1.x,
    		    	    dims.m1.y,
    		    	    dims.reduction_tile.z
    		};
	if(type == PROFILE_X)
	{
		return
		{
			dims.m1.x,
			dims.reduction_tile.y,
			dims.reduction_tile.z
		};
	}
	if(type == PROFILE_Y)
	{
		return
		{
			dims.reduction_tile.x,
			dims.m1.y,
			dims.reduction_tile.z
		};
	}
	if(type == PROFILE_Z)
	{
		return
		{
			dims.reduction_tile.x,
			dims.reduction_tile.y,
			dims.m1.z
		};
	}
	return dims.m1;
}

size_t
get_profile_reduce_scratchpad_size(const int profile, const VertexBufferArray vba)
{
	if(!reduced_profiles[profile]) return 0;
	const auto dims = acGetProfileReduceScratchPadDims(profile,vba.profile_dims[profile]);
	return dims.x*dims.y*dims.z*sizeof(AcReal);
}


void
init_scratchpads(VertexBufferArray* vba)
{
    vba->scratchpad_states = (AcScratchpadStates*)malloc(sizeof(AcScratchpadStates));
    memset(vba->scratchpad_states,0,sizeof(AcScratchpadStates));
    // Reductions
    {
	//TP: this is dangerous since it is not always true for DSL reductions but for now keep it
    	for(int i = 0; i < NUM_REAL_SCRATCHPADS; ++i) {
	    const size_t bytes =  
		    		  (i >= NUM_REAL_OUTPUTS) ? get_profile_reduce_scratchpad_size(i-NUM_REAL_OUTPUTS,*vba) :
				  0;
	    allocate_scratchpad_real(i,bytes,vba->scratchpad_states->reals[i]);
	    if(i < NUM_REAL_OUTPUTS)
	    {
	    	vba->reduce_buffer_real[i].src = &d_reduce_scratchpads_real[i];
	    	vba->reduce_buffer_real[i].cub_tmp = (AcReal**)malloc(sizeof(AcReal*));
	    	*(vba->reduce_buffer_real[i].cub_tmp) = NULL;
	    	vba->reduce_buffer_real[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    	*(vba->reduce_buffer_real[i].cub_tmp_size) = 0;

	    	vba->reduce_buffer_real[i].buffer_size    = &d_reduce_scratchpads_size_real[i];
    		device_malloc((void**) &vba->reduce_buffer_real[i].res,sizeof(AcReal));
	    }
	    else
	    {
		    const Profile prof = (Profile)(i-NUM_REAL_OUTPUTS);
		    const auto dims = acGetProfileReduceScratchPadDims(prof,vba->profile_dims[prof]);
		    vba->profile_reduce_buffers[prof].src = 
		    {
			    d_reduce_scratchpads_real[i],
			    dims.x*dims.y*dims.z,
			    true,
			    (AcShape) { dims.x,dims.y,dims.z,1}
		    };
		    vba->profile_reduce_buffers[prof].transposed = acBufferCreateTransposed(
				vba->profile_reduce_buffers[prof].src, 
				acGetMeshOrderForProfile(prof_types[prof])
				  );
		    vba->profile_reduce_buffers[prof].mem_order = acGetMeshOrderForProfile(prof_types[prof]);

	    	    vba->profile_reduce_buffers[prof].cub_tmp = (AcReal**)malloc(sizeof(AcReal*));
	    	    *(vba->profile_reduce_buffers[prof].cub_tmp) = NULL;
	    	    vba->profile_reduce_buffers[prof].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    	    *(vba->profile_reduce_buffers[prof].cub_tmp_size) = 0;
	    }
    	}
    }
    {
    	for(int i = 0; i < NUM_INT_OUTPUTS; ++i) {
	    const size_t bytes = 0;
	    allocate_scratchpad_int(i,bytes,vba->scratchpad_states->ints[i]);

	    vba->reduce_buffer_int[i].src= &d_reduce_scratchpads_int[i];
	    vba->reduce_buffer_int[i].cub_tmp = (int**)malloc(sizeof(int*));
	    *(vba->reduce_buffer_int[i].cub_tmp) = NULL;
	    vba->reduce_buffer_int[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    *(vba->reduce_buffer_int[i].cub_tmp_size) = 0;
	    vba->reduce_buffer_int[i].buffer_size    = &d_reduce_scratchpads_size_int[i];
    	    device_malloc((void**) &vba->reduce_buffer_int[i].res,sizeof(int));
    	}

#if AC_DOUBLE_PRECISION
    	for(int i = 0; i < NUM_FLOAT_OUTPUTS; ++i) {
	    const size_t bytes = 0;
	    allocate_scratchpad_float(i,bytes,vba->scratchpad_states->floats[i]);

	    vba->reduce_buffer_float[i].src= &d_reduce_scratchpads_float[i];
	    vba->reduce_buffer_float[i].cub_tmp = (float**)malloc(sizeof(float*));
	    *(vba->reduce_buffer_float[i].cub_tmp) = NULL;
	    vba->reduce_buffer_float[i].cub_tmp_size = (size_t*)malloc(sizeof(size_t));
	    *(vba->reduce_buffer_float[i].cub_tmp_size) = 0;
	    vba->reduce_buffer_float[i].buffer_size    = &d_reduce_scratchpads_size_float[i];
    	    device_malloc((void**) &vba->reduce_buffer_float[i].res,sizeof(float));
    	}
#endif
    }
}
static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info)
{
   #include "user_builtin_non_scalar_constants.h"
   const Volume n0 = to_volume(info[AC_nmin]);
   const Volume n1 = to_volume(info[AC_nlocal_max]);
   const Volume m0 = (Volume){0, 0, 0};
   const Volume m1 = to_volume(info[AC_mlocal]);
   const Volume nn = to_volume(info[AC_nlocal]);
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

static inline AcMeshDims
acGetMeshDims(const AcMeshInfo info, const VertexBufferHandle vtxbuf)
{
   #include "user_builtin_non_scalar_constants.h"
   const Volume n0 = to_volume(info[AC_nmin]);
   const Volume m1 = to_volume(info[vtxbuf_dims[vtxbuf]]);
   const Volume n1 = m1-n0;
   const Volume m0 = (Volume){0, 0, 0};
   const Volume nn = m1-n0*2;
   const Volume reduction_tile = (Volume)
   {
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].x),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].y),
	   as_size_t(info.int3_params[AC_reduction_tile_dimensions].z)
   };

   return (AcMeshDims){
       .n0 = n0,
       .n1 = n1,
       .m0 = m0,
       .m1 = m1,
       .nn = nn,
       .reduction_tile = reduction_tile,
   };
}

AcReal* vba_in_buff = NULL;
AcReal* vba_out_buff = NULL;

VertexBufferArray
acVBACreate(const AcMeshInfo config)
{
  //TP: !HACK!
  //TP: Get active dimensions at the time VBA is created, works for now but should be moved somewhere else
  #include "user_builtin_non_scalar_constants.h"
  dimension_inactive = config[AC_dimension_inactive];
  sparse_autotuning  = config[AC_sparse_autotuning];
  raytracing_subblock = config[AC_raytracing_block_factors];
  x_ray_shared_mem_block_size = config[AC_x_ray_shared_mem_block_size];
  z_ray_shared_mem_block_size = config[AC_z_ray_shared_mem_block_size];

  max_tpb_for_reduce_kernels = config[AC_max_tpb_for_reduce_kernels];
  VertexBufferArray vba;
  vba.on_device.block_factor = config[AC_thread_block_loop_factors];

  vba.computational_dims = acGetMeshDims(config);

  size_t in_bytes  = 0;
  size_t out_bytes = 0;
  for(int i = 0; i  < NUM_FIELDS; ++i)
  {
  	vba.dims[i]    = acGetMeshDims(config,Field(i));
  	size_t count = vba.dims[i].m1.x*vba.dims[i].m1.y*vba.dims[i].m1.z;
  	size_t bytes = sizeof(vba.on_device.in[0][0]) * count;
  	vba.counts[i]         = count;
  	vba.bytes[i]          = bytes;
	in_bytes  += vba.bytes[i];
	if(vtxbuf_is_auxiliary[i]) continue;
	out_bytes += vba.bytes[i];
  }
  for(int p = 0; p < NUM_PROFILES; ++p)
  {
	  vba.profile_dims[p] = acGetMeshDims(config);
  	  vba.profile_counts[p] = vba.profile_dims[p].m1.x*vba.profile_dims[p].m1.y*vba.profile_dims[p].m1.z;
  }
  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
  	size_t count = vba.computational_dims.m1.x*vba.computational_dims.m1.y*vba.computational_dims.m1.z;
	device_malloc(&vba.on_device.complex_in[field],sizeof(AcComplex)*count);
  }

  ERRCHK_ALWAYS(vba_in_buff == NULL);
  ERRCHK_ALWAYS(vba_out_buff == NULL);
  device_malloc((void**)&vba_in_buff,in_bytes);
  device_malloc((void**)&vba_out_buff,out_bytes);

  size_t out_offset = 0;
  size_t in_offset = 0;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.on_device.in[i] = vba_in_buff + in_offset;
    ERRCHK_ALWAYS(vba.on_device.in[i] != NULL);
    in_offset += vba.counts[i];
    //device_malloc((void**) &vba.on_device.out[i],bytes);
    if (vtxbuf_is_auxiliary[i])
    {
      vba.on_device.out[i] = vba.on_device.in[i];
      ERRCHK_ALWAYS(vba.on_device.out[i] != NULL);
    }else{
      vba.on_device.out[i] = (vba_out_buff + out_offset);
      out_offset += vba.counts[i];
      if(vba.on_device.out[i] == NULL)
      {
         fprintf(stderr,"In bytes %zu; Out bytes: %zu\n",in_bytes,out_bytes);	
	 fflush(stderr);
       	 ERRCHK_ALWAYS(vba.on_device.out[i] != NULL);
      }
    }
  }


  AcArrayTypes::run<allocate_arrays>(config);

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  vba.on_device.profiles = acPBACreate(vba.profile_dims);
  init_scratchpads(&vba);

  acVBAReset(0, &vba);
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
  return vba;
}

template <typename P>
struct update_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array : get_params<P>())
		{
			if (is_dconst(array) || !is_alive(array)) continue;
			auto config_array = config[array];
			auto gmem_array   = get_empty_pointer(array);
			memcpy_from_gmem_array(array,gmem_array);
			size_t bytes = sizeof(config_array[0])*get_array_length(array,config);
			if (config_array == nullptr && gmem_array != nullptr) 
				device_free(&gmem_array,bytes);
			else if (config_array != nullptr && gmem_array  == nullptr) 
				device_malloc(&gmem_array,bytes);
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};
void
acUpdateArrays(const AcMeshInfo config)
{
  AcArrayTypes::run<update_arrays>(config);
}

template <typename P>
struct free_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array: get_params<P>())
		{
			auto config_array = config[array];
			if (config_array == nullptr || is_dconst(array) || !is_alive(array)) continue;
			auto gmem_array = get_empty_pointer(array);
			memcpy_from_gmem_array(array,gmem_array);
			device_free(&gmem_array, get_array_length(array,config));
			memcpy_to_gmem_array(array,gmem_array);
		}
	}
};
void
destroy_profiles(VertexBufferArray* vba)
{
    for(int i = 0; i < NUM_PROFILES; ++i)
    {
        //TP: will break if allocated with compressed memory but too lazy to fix now: :(
        device_free((void**)&(vba->profile_reduce_buffers[i].transposed),0);
        free_scratchpad_real(i+NUM_REAL_OUTPUTS);
    }
}
void
destroy_real_scratchpads(VertexBufferArray* vba)
{
    for(int j = 0; j < NUM_REAL_OUTPUTS; ++j)
    {
	free_scratchpad_real(j);
	vba->reduce_buffer_real[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_real[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_real[j].res));

	free(vba->reduce_buffer_real[j].cub_tmp);
	free(vba->reduce_buffer_real[j].cub_tmp_size);
    }
}

void
destroy_scratchpads(VertexBufferArray* vba)
{
    destroy_real_scratchpads(vba);

    destroy_profiles(vba);

    for(int j = 0; j < NUM_INT_OUTPUTS; ++j)
    {
	free_scratchpad_int(j);
	vba->reduce_buffer_int[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_int[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_int[j].res));

	free(vba->reduce_buffer_int[j].cub_tmp);
	free(vba->reduce_buffer_int[j].cub_tmp_size);
    }
#if AC_DOUBLE_PRECISION
    for(int j = 0; j < NUM_FLOAT_OUTPUTS; ++j)
    {
	free_scratchpad_float(j);
	vba->reduce_buffer_float[j].src = NULL;

        ERRCHK_CUDA_ALWAYS(acFree(*vba->reduce_buffer_float[j].cub_tmp));
        ERRCHK_CUDA_ALWAYS(acFree(vba->reduce_buffer_float[j].res));

	free(vba->reduce_buffer_float[j].cub_tmp);
	free(vba->reduce_buffer_float[j].cub_tmp_size);
    }
#endif
}

void
acVBADestroy(VertexBufferArray* vba, const AcMeshInfo config)
{
  destroy_scratchpads(vba);
  //TP: does not work for compressible memory TODO: fix it if needed
  device_free(&(vba_in_buff), 0);
  device_free(&(vba_out_buff), 0);
  for(int field = 0; field < NUM_COMPLEX_FIELDS; ++field)
  {
  	device_free(&vba->on_device.complex_in[field], 0);
  }

  //Free arrays
  AcArrayTypes::run<free_arrays>(config);
  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBADestroy(&vba->on_device.profiles,vba->profile_dims);
  memset(vba->profile_dims,0,NUM_PROFILES*sizeof(vba->profile_dims[0]));
  memset(vba->bytes,0,NUM_ALL_FIELDS*sizeof(size_t));
  memset(vba->dims,0,NUM_ALL_FIELDS*sizeof(vba->dims[0]));
}



int
get_num_of_warps(const dim3 bpg, const dim3 tpb)
{
	const size_t warp_size = get_device_prop().warpSize;
	const int num_of_warps_per_block = (tpb.x*tpb.y*tpb.z + warp_size-1)/warp_size;
	const int num_of_blocks = bpg.x*bpg.y*bpg.z;
	return num_of_warps_per_block*num_of_blocks;
}

int
get_current_device()
{
	int device{};
	ERRCHK_CUDA_ALWAYS(acGetDevice(&device));
	return device;
}

bool
supports_cooperative_launches()
{
	static bool called{};
	static int supportsCoopLaunch{};
	if(called)
	{
		ERRCHK_ALWAYS(supportsCoopLaunch);
		return bool(supportsCoopLaunch);
	}
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, get_current_device());
	called = true;
	return bool(supportsCoopLaunch);
}
void
launch_kernel(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba, const dim3 bpg, const dim3 tpb, const size_t smem, const cudaStream_t stream)
{
  if(is_coop_raytracing_kernel(kernel) && supports_cooperative_launches())
  {
	void* args[] = {(void*)&start,(void*)&end,(void*)&vba.on_device};
	cudaLaunchCooperativeKernel((void*)kernels[kernel],bpg,tpb,args,smem,stream);
  }
  else
  {
  	KERNEL_VBA_LAUNCH(kernels[kernel],bpg,tpb,smem,stream)(start,end,vba.on_device);
  }
}
void
launch_kernel(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba, const dim3 bpg, const dim3 tpb, const size_t smem)
{
	launch_kernel(kernel,start,end,vba,bpg,tpb,smem,0);
}


const Volume 
get_kernel_end(const AcKernel kernel, const Volume start, const Volume end)
{
	if(is_raytracing_kernel(kernel))
	{
		const auto step_direction = raytracing_step_direction(kernel);
		if(step_direction.z) return (Volume){end.x,end.y,start.z+1};
		if(step_direction.y) return (Volume){end.x,start.y+1,end.z};
		if(step_direction.x) return (Volume){start.x+1,end.y,end.z};
	}
	return (Volume){end.x,end.y,end.z};

}
AcResult
acLaunchKernel(AcKernel kernel, const cudaStream_t stream, const Volume start_volume,
               const Volume end_volume, VertexBufferArray vba)
{
  const int3 start = to_int3(start_volume);
  const int3 end   = to_int3(get_kernel_end(kernel,start_volume,end_volume));

  const TBConfig tbconf = getOptimalTBConfig(kernel, start, end, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims),kernel,vba.on_device.block_factor, to_volume(tpb)));
  const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));
  if (kernel_calls_reduce[kernel] && reduce_offsets[kernel].find(start) == reduce_offsets[kernel].end())
  {
  	reduce_offsets[kernel][start] = kernel_running_reduce_offsets[kernel];
  	kernel_running_reduce_offsets[kernel] += get_num_of_warps(bpg,tpb);
	resize_scratchpads_to_fit(kernel_running_reduce_offsets[kernel],vba,kernel);
  }

  if(kernel_calls_reduce[kernel]) vba.on_device.reduce_offset = reduce_offsets[kernel][start];
  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
  launch_kernel(kernel,start,end,vba,bpg,tpb,smem,stream);
  ERRCHK_CUDA_KERNEL();

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}

AcResult
acBenchmarkKernel(AcKernel kernel, const int3 start, const int3 end,
                  VertexBufferArray vba)
{
  const TBConfig tbconf = getOptimalTBConfig(kernel, start, end, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));
  const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));

  // Timer create
  cudaEvent_t tstart, tstop;
  ERRCHK_CUDA(cudaEventCreate(&tstart));
  ERRCHK_CUDA(cudaEventCreate(&tstop));

  // Warmup
  ERRCHK_CUDA(cudaEventRecord(tstart));
  KERNEL_LAUNCH(kernels[kernel],bpg, tpb, smem)(start, end, vba.on_device);
  ERRCHK_CUDA(cudaEventRecord(tstop));
  ERRCHK_CUDA(cudaEventSynchronize(tstop));
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());

  // Benchmark
  ERRCHK_CUDA(cudaEventRecord(tstart)); // Timing start
  KERNEL_LAUNCH(kernels[kernel],bpg,tpb,smem)(start, end, vba.on_device);
  ERRCHK_CUDA(cudaEventRecord(tstop)); // Timing stop
  ERRCHK_CUDA(cudaEventSynchronize(tstop));
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, tstart, tstop);

  ERRCHK_ALWAYS(kernel < NUM_KERNELS);
  printf("Kernel %s time elapsed: %g ms\n", kernel_names[kernel],
         static_cast<double>(milliseconds));

  // Timer destroy
  ERRCHK_CUDA(cudaEventDestroy(tstart));
  ERRCHK_CUDA(cudaEventDestroy(tstop));

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}


AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t /* stream */,
              const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  // Note important acDeviceSynchronize below
  //
  // Constant memory allocated for stencils is shared among kernel
  // invocations, therefore a race condition is possible when updating
  // the coefficients. To avoid this, all kernels that can access
  // the coefficients must be completed before starting async copy to
  // constant memory
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());

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
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());

  const size_t bytes = sizeof(data[0][0][0]) * STENCIL_DEPTH * STENCIL_HEIGHT *
                       STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyFromSymbol(
      data, stencils, bytes, stencil * bytes, cudaMemcpyDeviceToHost);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};

AcResult
acLoadRealReduceRes(cudaStream_t stream, const AcRealOutputParam param, const AcReal* value)
{
  	const size_t offset =   (size_t)(&d_reduce_real_res_symbol[param]) - (size_t)&d_reduce_real_res_symbol;
	ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_reduce_real_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
	return AC_SUCCESS;
}

AcResult
acLoadIntReduceRes(cudaStream_t stream, const AcIntOutputParam param, const int* value)
{
  	const size_t offset =   (size_t)(&d_reduce_int_res_symbol[param]) - (size_t)&d_reduce_int_res_symbol;
	ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_reduce_int_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
	return AC_SUCCESS;
}

#if AC_DOUBLE_PRECISION
AcResult
acLoadFloatReduceRes(cudaStream_t stream, const AcFloatOutputParam param, const float* value)
{
  	const size_t offset =   (size_t)&d_reduce_float_res_symbol[param]- (size_t)&d_reduce_float_res_symbol;
	ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_reduce_float_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
	return AC_SUCCESS;
}
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
  	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize()); /* See note in acLoadStencil */

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
	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
	ERRCHK_ALWAYS(values  != nullptr);
	const size_t bytes = length*sizeof(values[0]);
	if (!is_dconst(array))
	{
		if (!is_alive(array)) return AC_NOT_ALLOCATED;
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
	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
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
    if(progress == 0)
    	fprintf(stream,"] %d%%  ", progress);
    else if(progress != 100)
    	fprintf(stream,"] %d%% ", progress);
    else
    	fprintf(stream,"] %d%%", progress);
}
void
printAutotuningStatus(const AcKernel kernel, const float best_time, const int progress)
{
   if(grid_pid != 0) return;
   fprintf(stderr,"\nAutotuning %s ",kernel_names[kernel]);
   printProgressBar(stderr,progress);
   if(best_time != INFINITY) fprintf(stderr," %14e",(double)best_time);
   if (progress == 100) fprintf(stderr,"\n");
   fflush(stderr);
}

void
logAutotuningStatus(const size_t counter, const size_t num_samples, const AcKernel kernel, const float best_time)
{
    const AcReal percent_of_num_samples = AcReal(num_samples)/100.0;
    for (size_t progress = 0; progress <= 90; ++progress)
    {
	      if (counter == floor(percent_of_num_samples*progress)  && (progress % 10 == 0))
	      {
		        printAutotuningStatus(kernel,best_time,progress);
	      }
    }
}

static AcAutotuneMeasurement
gather_best_measurement(const AcAutotuneMeasurement local_best)
{
	return gather_func(local_best);
}

void
make_vtxbuf_input_params_safe(VertexBufferArray& vba, const AcKernel kernel)
{
  //TP: have to set reduce offset zero since it might not be
  vba.on_device.reduce_offset = 0;
//#include "safe_vtxbuf_input_params.h"
}
int3
get_kernel_dims(const AcKernel kernel, const int3 start, const int3 end)
{
  return is_coop_raytracing_kernel(kernel) ? ceil_div(end-start,raytracing_subblock) : end-start;
}

static TBConfig
autotune(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba)
{
  const int3 dims = get_kernel_dims(kernel,start,end);
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


  //TP: since autotuning should be quite fast when the dim is not NGHOST only log for actually 3d portions
  const bool builtin_kernel = strlen(kernel_names[kernel]) > 2 && kernel_names[kernel][0] == 'A' && kernel_names[kernel][1] == 'C';
  const bool large_launch   = is_large_launch(dims);
  const bool log = is_raytracing_kernel(kernel) || (!builtin_kernel && large_launch);

  AcAutotuneMeasurement best_measurement = {INFINITY,(dim3){0,0,0}};
  const int num_iters = 2;

  // Get device hardware information
  const auto props = get_device_prop();
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

  int3 tpb_end = dims; 
  if(is_raytracing_kernel(kernel))
  {
	const auto dir = raytracing_step_direction(kernel);
	if(dir.z) tpb_end.z = 1;
	if(dir.y) tpb_end.y = 1;
	if(dir.x) tpb_end.x = 1;
  }
  //TP: emprically and thinking about it y and z usually cannot be too big (since x is usually quite large) so we can limit then when performing sparse autotuning
  if(sparse_autotuning)
  {
	  if(!raytracing_step_direction(kernel).x)
	  {
	  	tpb_end.y = min(tpb_end.y,32);
	  }
	  tpb_end.z = min(tpb_end.z,32);
  }
  const int x_increment = min(
		  			minimum_transaction_size_in_elems,
					tpb_end.x
		            );


  std::vector<int3> samples{};
  for (int z = 1; z <= min(max_threads_per_block,tpb_end.z); ++z) {
    for (int y = 1; y <= min(max_threads_per_block,tpb_end.y); ++y) {
      for (int x = x_increment;
           x <= min(max_threads_per_block,tpb_end.x); x += x_increment) {


        if (x * y * z > max_threads_per_block)
          break;
        const dim3 tpb(x, y, z);
        const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        if (smem > max_smem)
          continue;

        //if ((x * y * z) % props.warpSize && (x*y*z) >props.warpSize)
        //  continue;

        if (!is_valid_configuration(to_volume(dims), to_volume(tpb),kernel))
          continue;
	//TP: should be emplace back but on my laptop the CUDA compiler gives a cryptic error message that I do not care to debug
        samples.push_back((int3){x,y,z});
      }
    }
  }
  if(samples.size() == 0)
  {
	fprintf(stderr,"Found no suitable thread blocks for Kernel %s!\n",kernel_names[kernel]);
	fflush(stderr);
  	ERRCHK_ALWAYS(samples.size() > 0);
  }
  size_t counter  = 0;
  size_t start_samples{};
  size_t end_samples{};

  const bool on_halos =
	  (start.x < (int)vba.computational_dims.n0.x) ||
	  (start.y < (int)vba.computational_dims.n0.y) ||
	  (start.z < (int)vba.computational_dims.n0.z) ||
                          
	  (end.x >=  (int)vba.computational_dims.n1.x) ||
	  (end.y >=  (int)vba.computational_dims.n1.y) ||
	  (end.z >=  (int)vba.computational_dims.n1.z);

  const bool parallel_autotuning = !on_halos && AC_MPI_ENABLED;
  if(parallel_autotuning)
  {
  	const size_t portion = ceil_div(samples.size(),nprocs);
  	start_samples = portion*grid_pid;
  	end_samples   = min(samples.size(), portion*(grid_pid+1));
  }
  else
  {
  	start_samples = 0;
  	end_samples   = samples.size();
  }
  const size_t n_samples = end_samples-start_samples;

  //TP: logs the percent 0% which is useful to know the autotuning has started
  if (log) logAutotuningStatus(counter,n_samples,kernel,best_measurement.time / num_iters);
  for(size_t sample  = start_samples; sample < end_samples; ++sample)
  {
        auto x = samples[sample].x;
        auto y = samples[sample].y;
        auto z = samples[sample].z;
        const dim3 tpb(x, y, z);
        const dim3 bpg    = to_dim3(
                                get_bpg(to_volume(dims),kernel,vba.on_device.block_factor,
                                to_volume(tpb)
                                ));
	const int n_warps = get_num_of_warps(bpg,tpb);
	if(kernel_calls_reduce[kernel])
		resize_scratchpads_to_fit(n_warps,vba,kernel);
        const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        cudaEvent_t tstart, tstop;
        ERRCHK_CUDA(cudaEventCreate(&tstart));
        ERRCHK_CUDA(cudaEventCreate(&tstop));

        launch_kernel(kernel,start,end,vba,bpg,tpb,smem);
        ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
        ERRCHK_CUDA(cudaEventRecord(tstart)); // Timing start
        for (int i = 0; i < num_iters; ++i)
	{
        	launch_kernel(kernel,start,end,vba,bpg,tpb,smem);
	}
        ERRCHK_CUDA(cudaEventRecord(tstop)); // Timing stop
        ERRCHK_CUDA(cudaEventSynchronize(tstop));

        float milliseconds = 0;
        ERRCHK_CUDA(cudaEventElapsedTime(&milliseconds, tstart, tstop));

        ERRCHK_CUDA(cudaEventDestroy(tstart));
        ERRCHK_CUDA(cudaEventDestroy(tstop));
        ++counter;
        if (log) logAutotuningStatus(counter,n_samples,kernel,best_measurement.time / num_iters);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        const auto err = cudaGetLastError();
        //TP: it is fine to simply skip invalid configuration values since it can be because of too large tpb's
        //We simply do not count them for finding the optim config
        if(err == cudaErrorInvalidConfiguration) continue;
        if(err == cudaErrorLaunchOutOfResources) continue;
        if (err != cudaSuccess) {
          //TP: reset autotune results
          fprintf(stderr,"\nFailed while autotuning: %s\nReason: %s\n",kernel_names[kernel],cudaGetErrorName(err));
          FILE* fp = fopen(autotune_csv_path,"w");
          fclose(fp);
          ERRCHK_ALWAYS(err == cudaSuccess);
        }

        if (milliseconds < best_measurement.time) {
          best_measurement.time = milliseconds;
          best_measurement.tpb = tpb;
        }

        // printf("Auto-optimizing... Current tpb: (%d, %d, %d), time %f ms\n",
        //        tpb.x, tpb.y, tpb.z, (double)milliseconds / num_iters);
        // fflush(stdout);
  }
  best_measurement =  parallel_autotuning ? gather_best_measurement(best_measurement) : best_measurement;
  if(log) printAutotuningStatus(kernel,best_measurement.time/num_iters,100);
  c.tpb = best_measurement.tpb;
  if(grid_pid == 0)
  {
        FILE* fp = fopen(autotune_csv_path, "a");
        ERRCHK_ALWAYS(fp);
#if IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
        fprintf(fp, "%d, (%d, %d, %d), (%d, %d, %d), %g\n", IMPLEMENTATION, nx, ny,
                nz, best_measurement.tpb.x, best_measurement.tpb.y, best_measurement.tpb.z,
                (double)best_measurement.time / num_iters);
#else
        fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %d, %g, %s, %d, %d, %d, %d, %d, %d, %d\n", IMPLEMENTATION, kernel, dims.x,
                dims.y, dims.z, best_measurement.tpb.x, best_measurement.tpb.y, best_measurement.tpb.z,
                (double)best_measurement.time / num_iters, kernel_names[kernel],
		vba.on_device.block_factor.x,vba.on_device.block_factor.y,vba.on_device.block_factor.z
		,raytracing_subblock.x
		,raytracing_subblock.y
		,raytracing_subblock.z
		,sparse_autotuning
		);
#endif
        fclose(fp);
	fflush(fp);
  }
  if (c.tpb.x * c.tpb.y * c.tpb.z <= 0) {
    fprintf(stderr,
            "Fatal error: failed to find valid thread block dimensions for (%d,%d,%d) launch of %s.\n"
            ,dims.x,dims.y,dims.z,kernel_names[kernel]);
  }
  ERRCHK_ALWAYS(c.tpb.x * c.tpb.y * c.tpb.z > 0);
  //TP: done to ensure scratchpads are reset after autotuning
  if(vba.scratchpad_states) memset(vba.scratchpad_states,0,sizeof(AcScratchpadStates));
  return c;
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

int3
acReadOptimTBConfig(const AcKernel kernel, const int3 dims, const int3 block_factors)
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
	  if(entry.size == 17)
      	  {
      	     int kernel_index  = atoi(entry.data[1]);
      	     int3 read_dims = {atoi(entry.data[2]), atoi(entry.data[3]), atoi(entry.data[4])};
      	     int3 tpb = {atoi(entry.data[5]), atoi(entry.data[6]), atoi(entry.data[7])};
      	     double time = atof(entry.data[8]);
      	     int3 read_block_factors = {atoi(entry.data[10]), atoi(entry.data[11]), atoi(entry.data[12])};
      	     int3 read_raytracing_factors = {atoi(entry.data[13]), atoi(entry.data[14]), atoi(entry.data[15])};
	     int  was_sparse = atoi(entry.data[16]);
      	     if(time < best_time && kernel_index == kernel && read_dims == dims && read_block_factors == block_factors && read_raytracing_factors == raytracing_subblock && was_sparse == sparse_autotuning)
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
getOptimalTBConfig(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba)
{
  const int3 dims = get_kernel_dims(kernel,start,end);
  for (auto c : tbconfigs)
    if (c.kernel == kernel && c.dims == dims)
      return c;

  const int3 read_tpb = acReadOptimTBConfig(kernel,dims,vba.on_device.block_factor);
  TBConfig c  = (read_tpb != (int3){-1,-1,-1})
          ? (TBConfig){kernel,dims,(dim3){(uint32_t)read_tpb.x, (uint32_t)read_tpb.y, (uint32_t)read_tpb.z}}
          : autotune(kernel,start,end,vba);
  tbconfigs.push_back(c);
  return c;
}

AcKernel
acGetOptimizedKernel(const AcKernel kernel_enum, const VertexBufferArray vba)
{
	#include "user_kernels_ifs.h"
	//silence unused warnings
	(void)vba;
	return kernel_enum;
	//return kernels[(int) kernel_enum];
}
void
acVBASwapBuffer(const Field field, VertexBufferArray* vba)
{
  AcReal* tmp     = vba->on_device.in[field];
  vba->on_device.in[field]  = vba->on_device.out[field];
  vba->on_device.out[field] = tmp;
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
  AcReal* tmp                = vba->on_device.profiles.in[profile];
  vba->on_device.profiles.in[profile]  = vba->on_device.profiles.out[profile];
  vba->on_device.profiles.out[profile] = tmp;
}

void
acPBASwapBuffers(VertexBufferArray* vba)
{
  for (int i = 0; i < NUM_PROFILES; ++i)
    acPBASwapBuffer((Profile)i, vba);
}

template <typename P>
struct load_all_scalars_uniform
{
	void operator()(const AcMeshInfo& config)
	{
		for(P i : get_params<P>())
			acLoadUniform(0,  i, config[i]);
	}
};

template <typename P>
struct load_all_arrays_uniform
{
	void operator()(const AcMeshInfo& config)
	{
		for(const P array : get_params<P>())
		{
			auto config_array = config[array];
      			if (config_array != nullptr)
				acLoadArrayUniform(array,config_array, get_array_length(array,config));
		}
	}
};

AcResult
acLoadMeshInfo(const AcMeshInfo info, const cudaStream_t)
{
  /* See note in acLoadStencil */
  ERRCHK_CUDA(acDeviceSynchronize());
  AcResult retval = AC_SUCCESS;
  AcScalarTypes::run<load_all_scalars_uniform>(info);
  AcArrayTypes::run<load_all_arrays_uniform>(info);
  return retval;
}

//---------------
// static __host__ __device__ constexpr size_t
// acShapeSize(const AcShape& shape)
size_t
acShapeSize(const AcShape shape)
{
  return shape.x * shape.y * shape.z * shape.w;
}

__host__ __device__ constexpr bool
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

__host__ __device__ constexpr AcIndex
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

__host__ __device__ constexpr AcIndex
to_spatial(const size_t i, const AcShape& shape)
{
  return (AcIndex){
      .x = i % shape.x,
      .y = (i / shape.x) % shape.y,
      .z = (i / (shape.x * shape.y)) % shape.z,
      .w = i / (shape.x * shape.y * shape.z),
  };
}

__host__ __device__ constexpr size_t
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
  const AcIndex idx = to_spatial(
      static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x, block_shape);

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

typedef struct
{
	size_t x;
	size_t y;
} size_t2;

struct size_t2Hash {
    std::size_t operator()(const size_t2& v) const {
        return std::hash<size_t>()(v.x) ^ std::hash<size_t>()(v.y) << 1;
    }
};

std::unordered_map<size_t2,size_t*,size_t2Hash> segmented_reduce_offsets{};

static HOST_DEVICE_INLINE bool
operator==(const size_t2& a, const size_t2& b)
{
  return a.x == b.x && a.y == b.y;
}

//TP: will return a cached allocation if one is found
size_t*
get_offsets(const size_t count, const size_t num_segments)
{
  const size_t2 key = {count,num_segments};
  if(segmented_reduce_offsets.find(key) != segmented_reduce_offsets.end())
	  return segmented_reduce_offsets[key];

  size_t* offsets = (size_t*)malloc(sizeof(offsets[0]) * (num_segments + 1));
  ERRCHK_ALWAYS(num_segments > 0);
  ERRCHK_ALWAYS(offsets);
  ERRCHK_ALWAYS(count % num_segments == 0);
  for (size_t i = 0; i <= num_segments; ++i) {
    offsets[i] = i * (count / num_segments);
    ERRCHK_ALWAYS(offsets[i] <= count);
  }
  size_t* d_offsets = NULL;
  ERRCHK_CUDA_ALWAYS(cudaMalloc(&d_offsets, sizeof(d_offsets[0]) * (num_segments + 1)));
  ERRCHK_ALWAYS(d_offsets);
  ERRCHK_CUDA(cudaMemcpy(d_offsets, offsets, sizeof(d_offsets[0]) * (num_segments + 1),cudaMemcpyHostToDevice));
  free(offsets);
  segmented_reduce_offsets[key] = d_offsets;
  return d_offsets;
}


AcResult
acSegmentedReduce(const cudaStream_t stream, const AcReal* d_in,
                  const size_t count, const size_t num_segments, AcReal* d_out, AcReal** tmp_buffer, size_t* tmp_size)
{

  size_t* d_offsets = get_offsets(count,num_segments);

  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  ERRCHK_CUDA(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_segments, d_offsets, d_offsets + 1,
                                  stream));

  *tmp_size = device_resize((void**)tmp_buffer,*tmp_size,temp_storage_bytes);
  ERRCHK_CUDA(cub::DeviceSegmentedReduce::Sum((void*)(*tmp_buffer), temp_storage_bytes, d_in,
                            d_out, num_segments, d_offsets, d_offsets + 1,
                            stream));
  ERRCHK_CUDA_KERNEL();
  return AC_SUCCESS;
}

template <typename T>
AcResult
acReduceBase(const cudaStream_t stream, const AcReduceOp reduce_op, T buffer, const size_t count)
{
  ERRCHK(*(buffer.buffer_size)/sizeof(*(buffer.src)[0]) >= count);
  ERRCHK(buffer.src   != NULL);
  ERRCHK(buffer.src   != NULL);

  AcDeviceTmpBuffer temp_storage{NULL,0};
  cub_reduce(temp_storage,stream,*(buffer.src),count,buffer.res,reduce_op);

  *buffer.cub_tmp_size = device_resize((void**)buffer.cub_tmp,*buffer.cub_tmp_size,temp_storage.bytes);
  temp_storage.data = (void*)(*buffer.cub_tmp);
  cub_reduce(temp_storage,stream,*(buffer.src),count,buffer.res,reduce_op);
  return AC_SUCCESS;
}

AcResult
acReduceReal(const cudaStream_t stream, const AcReduceOp op, const AcRealScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}

#if AC_DOUBLE_PRECISION
AcResult
acReduceFloat(const cudaStream_t stream, const AcReduceOp op, const AcFloatScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}
#endif

AcResult
acReduceInt(const cudaStream_t stream, const AcReduceOp op, const AcIntScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}

AcResult
acComplexToReal(const AcComplex* src, const size_t count, AcReal* dst)
{
  acLaunchKernelVariadic1d(AC_COMPLEX_TO_REAL,0,0,count,(AcComplex*)src,dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acRealToComplex(const AcReal* src, const size_t count, AcComplex* dst)
{
  acLaunchKernelVariadic1d(AC_REAL_TO_COMPLEX,0,0,count,(AcReal*)src,dst);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}


AcResult
acMultiplyInplaceComplex(const AcReal value, const size_t count, AcComplex* array)
{
  acLaunchKernelVariadic1d(AC_MULTIPLY_INPLACE_COMPLEX,0,0,count,value,array);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

AcResult
acMultiplyInplace(const AcReal value, const size_t count, AcReal* array)
{
  acLaunchKernelVariadic1d(AC_MULTIPLY_INPLACE,0,0,count,value,array);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(acDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}
#define TILE_DIM (32)

void __global__ 
transpose_xyz_to_zyx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
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
		(int)start.x + threadIdx.x + block_offset.x,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.z + threadIdx.x + block_offset.z,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.x + threadIdx.z + block_offset.x
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.z ||  out_vertexIdx.y >= end.y || out_vertexIdx.z >= end.x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.z*out_vertexIdx.y + dims.z*dims.y*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.z];
}
void __global__ 
transpose_xyz_to_zxy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
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
		(int) start.x + threadIdx.x + block_offset.x,
		(int) start.y + threadIdx.y + block_offset.y,
		(int) start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.z + threadIdx.x + block_offset.z,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.x + threadIdx.z + block_offset.x
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.z ||  out_vertexIdx.y >= end.y || out_vertexIdx.z >= end.x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.z*out_vertexIdx.z + dims.z*dims.x*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.z];
}
void __global__ 
transpose_xyz_to_yxz(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
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
		(int) start.x + threadIdx.x + block_offset.x,
		(int) start.y + threadIdx.y + block_offset.y,
		(int) start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.y + threadIdx.x + block_offset.y,
		(int)start.x + threadIdx.y + block_offset.x,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.y ||  out_vertexIdx.y >= end.x || out_vertexIdx.z >= end.z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.y*out_vertexIdx.y + dims.x*dims.y*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.y];
}
void __global__ 
transpose_xyz_to_yzx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
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
		(int)start.x +threadIdx.x + block_offset.x,
		(int)start.y +threadIdx.y + block_offset.y,
		(int)start.z +threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.y + threadIdx.x + block_offset.y,
		(int)start.x + threadIdx.y + block_offset.x,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.y ||  out_vertexIdx.y >= end.x || out_vertexIdx.z >= end.z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : 0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.y*out_vertexIdx.z + dims.y*dims.z*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.y];
}
void __global__ 
transpose_xyz_to_xzy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	const dim3 in_block_offset =
	{
		blockIdx.x*blockDim.x,
		blockIdx.y*blockDim.y,
		blockIdx.z*blockDim.z
	};

	const dim3 vertexIdx = 
	{
		(int)start.x + threadIdx.x + in_block_offset.x,
		(int)start.y + threadIdx.y + in_block_offset.y,
		(int)start.z + threadIdx.z + in_block_offset.z
	};

	const bool oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	if(oob) return;
	dst[vertexIdx.x + dims.x*vertexIdx.z + dims.x*dims.z*vertexIdx.y] 
		= src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)];
}
static AcResult
acTransposeXYZ_ZYX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const dim3 tpb = {32,1,32};
	const Volume sub_dims = end-start;
	const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zyx,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_ZXY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const dim3 tpb = {32,1,32};

	const Volume sub_dims = end-start;
	const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zxy,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_YXZ(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};

	const Volume sub_dims = end-start;
	const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yxz,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_YZX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};

	const Volume sub_dims = end-start;
	const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yzx,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_XZY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const dim3 tpb = {32,32,1};
	const Volume sub_dims = end-start;
	const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_xzy,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}
static AcResult
acTransposeXYZ_XYZ(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	const Volume sub_dims = end-start;
	const size_t bytes = sub_dims.x*sub_dims.y*sub_dims.z*sizeof(AcReal);
	src = &src[start.x + dims.x*start.y + dims.x*dims.y*start.z];
	dst = &dst[start.x + dims.x*start.y + dims.x*dims.y*start.z];
	ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
	return AC_SUCCESS;
}
AcResult
acTransposeWithBounds(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	switch(order)
	{
		case(XYZ):
			return acTransposeXYZ_XYZ(src,dst,dims,start,end,stream);
		case (XZY):
			return acTransposeXYZ_XZY(src,dst,dims,start,end,stream);
		case (YXZ):
			return acTransposeXYZ_YXZ(src,dst,dims,start,end,stream);
		case (YZX):
			return acTransposeXYZ_YZX(src,dst,dims,start,end,stream);
		case(ZXY):
			return acTransposeXYZ_ZXY(src,dst,dims,start,end,stream);
		case(ZYX):
			return acTransposeXYZ_ZYX(src,dst,dims,start,end,stream);
	}
        ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

AcResult
acTranspose(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const cudaStream_t stream)
{
	return acTransposeWithBounds(order,src,dst,dims,(Volume){0,0,0},dims,stream);
}

size_t
get_count(const AcShape shape)
{
	return shape.x*shape.y*shape.z*shape.w;
}
static AcResult
ac_flush_scratchpad(VertexBufferArray vba, const int variable, const AcType type, const AcReduceOp op)
{

	const int n_elems = 
				type == AC_REAL_TYPE ?  NUM_REAL_OUTPUTS :
				type == AC_PROF_TYPE ?  NUM_PROFILES     :
				type == AC_INT_TYPE  ?  NUM_INT_OUTPUTS  :
#if AC_DOUBLE_PRECISION
				type == AC_FLOAT_TYPE  ?  NUM_FLOAT_OUTPUTS  :
#endif
				0;
	ERRCHK_ALWAYS(variable < n_elems);
	const size_t counts = 
			type == AC_INT_TYPE  ? (*vba.reduce_buffer_int[variable].buffer_size)/sizeof(int) :
#if AC_DOUBLE_PRECISION
			type == AC_FLOAT_TYPE  ? (*vba.reduce_buffer_float[variable].buffer_size)/sizeof(float) :
#endif
			type == AC_REAL_TYPE ? (*vba.reduce_buffer_real[variable].buffer_size)/sizeof(AcReal) :
			type == AC_PROF_TYPE ? (get_count(vba.profile_reduce_buffers[variable].src.shape)) :
			0;

	if(type == AC_REAL_TYPE)
	{
		if constexpr (NUM_REAL_OUTPUTS == 0) return AC_FAILURE;
		AcReal* dst = *(vba.reduce_buffer_real[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_real(op));
	}
	else if(type == AC_PROF_TYPE)
	{
		if constexpr(NUM_PROFILES == 0) return AC_FAILURE;
		AcReal* dst = vba.profile_reduce_buffers[variable].src.data;
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_real(op));
	}
#if AC_DOUBLE_PRECISION
	else if(type == AC_FLOAT_TYPE)
	{
		if constexpr(NUM_FLOAT_OUTPUTS  == 0) return AC_FAILURE;
		float* dst = *(vba.reduce_buffer_float[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_float(op));
	}
#endif
	else
	{
		if constexpr (NUM_INT_OUTPUTS == 0) return AC_FAILURE;
		int* dst = *(vba.reduce_buffer_int[variable].src);
		acKernelFlush(0,dst,counts,get_reduce_state_flush_var_int(op));
	}
  	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
	return AC_SUCCESS;
}
static AcReduceOp*
get_reduce_buffer_states(const VertexBufferArray vba, const AcType type)
{
	return
#if AC_DOUBLE_PRECISION
			type == AC_FLOAT_TYPE  ? vba.scratchpad_states->floats :
#endif
			type == AC_INT_TYPE    ? vba.scratchpad_states->ints  :
			type == AC_REAL_TYPE   ? vba.scratchpad_states->reals :
			type == AC_PROF_TYPE   ? &vba.scratchpad_states->reals[NUM_REAL_OUTPUTS] :
			NULL;
}
static UNUSED AcReduceOp
get_reduce_buffer_state(const VertexBufferArray vba, const int variable, const AcType type)
{
	return get_reduce_buffer_states(vba,type)[variable];
}
AcResult
acPreprocessScratchPad(VertexBufferArray vba, const int variable, const AcType type,const AcReduceOp op)
{
	AcReduceOp* states = get_reduce_buffer_states(vba,type);
	if(states[variable] == op) return AC_SUCCESS;
	states[variable] = op;
	return ac_flush_scratchpad(vba,variable,type,op);
}

AcMeshOrder
acGetMeshOrderForProfile(const AcProfileType type)
{
    	switch(type)
    	{
    	        case(PROFILE_X):
    	    	    return ZYX;
    	        case(PROFILE_Y):
		    return XZY;
    	        case(PROFILE_Z):
			return XYZ;
    	        case(PROFILE_XY):
			return ZXY;
    	        case(PROFILE_XZ):
			return YXZ;
    	        case(PROFILE_YX):
			return ZYX;
    	        case(PROFILE_YZ):
			return XYZ;
    	        case(PROFILE_ZX):
			return YZX;
    	        case(PROFILE_ZY):
			return XZY;
		case(PROFILE_NONE):
			return XYZ;
    	}
	return XYZ;
};

#include "load_ac_kernel_params.h"

int
acVerifyMeshInfo(const AcMeshInfo info)
{
  int retval = 0;
  for (size_t i = 0; i < NUM_INT_PARAMS; ++i) {
    if (info.int_params[i] == INT_MIN) {
      retval = -1;
      fprintf(stderr, "--- Warning: [%s] uninitialized ---\n",
              intparam_names[i]);
    }
  }
  for (size_t i = 0; i < NUM_INT3_PARAMS; ++i) {
    if (info.int3_params[i].x == INT_MIN || info.int3_params[i].y == INT_MIN ||
        info.int3_params[i].z == INT_MIN) {
      retval = -1;
      fprintf(stderr, "--- Warning: [%s] uninitialized ---\n",
              int3param_names[i]);
    }
  }
  for (size_t i = 0; i < NUM_REAL_PARAMS; ++i) {
    if (isnan(info.real_params[i])) {
      retval = -1;
      fprintf(stderr, "--- Warning: [%s] uninitialized ---\n",
              realparam_names[i]);
    }
  }
  for (int i = 0; i < NUM_REAL3_PARAMS; ++i) {
    if (isnan(info.real3_params[i].x) || isnan(info.real3_params[i].y) ||
        isnan(info.real3_params[i].z)) {
      retval = -1;
      fprintf(stderr, "--- Warning: [%s] uninitialized ---\n",
              real3param_names[i]);
    }
  }
  return retval;
}

const AcKernel*
acGetKernels()
{
	return kernel_enums;
}

AcResult
acRuntimeQuit()
{
	tbconfigs.clear();
	for(int kernel = 0; kernel < NUM_KERNELS; ++kernel)
	{
		reduce_offsets[kernel].clear();
		kernel_running_reduce_offsets[kernel] = 0;
	}
	segmented_reduce_offsets.clear();
	return AC_SUCCESS;
}
#if AC_FFT_ENABLED


#if AC_USE_HIP
#if AC_DOUBLE_PRECISION
#define AC_FFT_PRECISION rocfft_precision_double
#else
#define AC_FFT_PRECISION rocfft_precision_single
#endif

#include <rocfft.h>

rocfft_plan_description 
get_data_layout(const Volume domain_size)
{
    //TP: not sure are the offsets for rocfft in bytes or in number of elements so prefer to do the offseting via pointer arithmetic myself
    size_t offsets[]  = {0,0,0};
    size_t strides[]  = {domain_size.x*domain_size.y,domain_size.x,1};
    size_t distance = domain_size.x*domain_size.y*domain_size.z;
    // Create plan description
    rocfft_plan_description desc = nullptr;
    rocfft_status status = rocfft_plan_description_create(&desc);
    ERRCHK_ALWAYS((status == rocfft_status_success));
    status = rocfft_plan_description_set_data_layout(
        desc,
        rocfft_array_type_complex_interleaved,  // in_array_type
        rocfft_array_type_complex_interleaved,  // out_array_type
	offsets,
	offsets,
	3,
	strides,
	distance,
	3,
	strides,
	distance
        );

    ERRCHK_ALWAYS((status == rocfft_status_success));
    return desc;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst) {
    rocfft_plan_description desc = get_data_layout(domain_size);
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    // Create plan
    rocfft_plan plan = nullptr;
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    rocfft_status status = rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_forward,
	AC_FFT_PRECISION,
        3,            // Dimensions
        lengths,      // lengths
        1,            // batch
        desc);        // description
    if (status != rocfft_status_success) return AC_FAILURE;

    // Create execution info
    rocfft_execution_info info = nullptr;
    status = rocfft_execution_info_create(&info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Execute
    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};
    status = rocfft_execute(plan, in_buffer, out_buffer, info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);

    // Scaling (just like CUFFT doesn't scale by default)
    size_t complex_domain_size = domain_size.x * domain_size.y * domain_size.z;
    const AcReal scale = 1.0 / (subdomain_size.x * subdomain_size.y * subdomain_size.z);
    acMultiplyInplaceComplex(scale, complex_domain_size, dst);

    return AC_SUCCESS;
}


AcResult
acFFTBackwardTransformC2C(const AcComplex* src,
                                 const Volume domain_size,
                                 const Volume subdomain_size,
                                 const Volume starting_point,
                                 AcComplex* dst) {
    // Create plan description
    rocfft_plan_description desc = get_data_layout(domain_size);
    // Create inverse plan
    rocfft_plan plan = nullptr;
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    rocfft_status status = rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_inverse,
	AC_FFT_PRECISION,
        3,           // Dimensions
        lengths,     // FFT size
        1,           // Batch size
        desc);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Create execution info
    rocfft_execution_info info = nullptr;
    status = rocfft_execution_info_create(&info);
    if (status != rocfft_status_success) return AC_FAILURE;

    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};

    status = rocfft_execute(plan, in_buffer, out_buffer, info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);

    return AC_SUCCESS;
}
AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*) {
	return AC_FAILURE;
}
#else
#include <cufftXt.h>
#include <cuComplex.h>

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
	    {                                                                                                          \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
		abort();                                                                                               \
	    }                                                                                                          \
    }
#endif  // CUFFT_CALL


// TODO: if the buffer on GPU would be properly padded:
// https://docs.nvidia.com/cuda/cufft/index.html#data-layout
// we could use in-place transformation and save one buffer allocation
// Padding as mentioned in the link: padded to (n/2 + 1) in the least significant dimension.
AcResult
acFFTForwardTransformSymmetricR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
    buffer = buffer + (starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z));
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)(subdomain_size.x / 2) + 1};
    
    cufftHandle plan_r2c{};
    CUFFT_CALL(cufftCreate(&plan_r2c));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_r2c, 3, dims,
        inembed, 1, inembed[0], // in case inembed and onembed not needed could be: nullptr, 1, 0
        onembed, 1, onembed[0], //                                                  nullptr, 1, 0
        CUFFT_D2Z, 1, &workspace_size));
    
    size_t orig_domain_size = inembed[0] * inembed[1] * inembed[2];
    size_t complex_domain_size = onembed[0] * onembed[1] * onembed[2];    
    
    cuDoubleComplex* transformed = reinterpret_cast<cuDoubleComplex*>(transformed_in);
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)buffer, transformed, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{1.0 / orig_domain_size};
    acMultiplyInplaceComplex(scale, complex_domain_size, transformed_in);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    buffer = buffer + starting_offset;
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)domain_size.z, (int)domain_size.y, (int)(domain_size.x)};
    
    cufftHandle plan_r2c{};
    CUFFT_CALL(cufftCreate(&plan_r2c));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_r2c, 3, dims,
        inembed, 1, inembed[0], // in case inembed and onembed not needed could be: nullptr, 1, 0
        onembed, 1, onembed[0], //                                                  nullptr, 1, 0
        CUFFT_Z2Z, 1, &workspace_size));
    
    size_t complex_domain_size = onembed[0] * onembed[1] * onembed[2];    
    
    cuDoubleComplex* transformed = reinterpret_cast<cuDoubleComplex*>(transformed_in + starting_offset);
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)buffer, transformed, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{1.0 / ( dims[0] * dims[1] * dims[2])};
    acMultiplyInplaceComplex(scale, complex_domain_size, transformed_in);
    return AC_SUCCESS;
}




AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    buffer = buffer + (starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z));
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)(((int)subdomain_size.x) / 2) + 1};
    
    cufftHandle plan_c2r{};
    CUFFT_CALL(cufftCreate(&plan_c2r));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_c2r, 3, dims,
        onembed, 1, onembed[0],
        inembed, 1, inembed[0],
        CUFFT_Z2D, 1, &workspace_size));
    const cuDoubleComplex* transformed = reinterpret_cast<const cuDoubleComplex*>(transformed_in);
    CUFFT_CALL(cufftXtExec(plan_c2r, (void*)transformed, buffer, CUFFT_INVERSE));
    CUFFT_CALL(cufftDestroy(plan_c2r));
    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcComplex* buffer) {
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    buffer = buffer + starting_offset;
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)domain_size.z, (int)domain_size.y, (int)(((int)domain_size.x))};
    
    cufftHandle plan_c2r{};
    CUFFT_CALL(cufftCreate(&plan_c2r));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_c2r, 3, dims,
        onembed, 1, onembed[0],
        inembed, 1, inembed[0],
        CUFFT_Z2Z, 1, &workspace_size));
    const cuDoubleComplex* transformed = reinterpret_cast<const cuDoubleComplex*>(transformed_in + starting_offset);
    CUFFT_CALL(cufftXtExec(plan_c2r, (void*)transformed, buffer, CUFFT_INVERSE));
    CUFFT_CALL(cufftDestroy(plan_c2r));
    return AC_SUCCESS;
}
#endif //AC_USE_HIP
AcResult
acFFTBackwardTransformC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* tmp = NULL;
    device_malloc(&tmp,bytes);
    acFFTBackwardTransformC2C(transformed_in,domain_size,subdomain_size,starting_point,tmp);
    acComplexToReal(tmp,count,buffer);
    device_free(&tmp,0);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* tmp = NULL;
    device_malloc(&tmp,bytes);
    acRealToComplex(buffer,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,transformed_in);
    device_free(&tmp,0);
    return AC_SUCCESS;
}

#else
AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTForwardTransformR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTBackwardTransformC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
#endif

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
acMemcpyAsync(AcReal* dst, const AcReal* src, const size_t bytes, cudaMemcpyKind kind, const cudaStream_t stream)
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

