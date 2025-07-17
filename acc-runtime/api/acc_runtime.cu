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

#include "device_details.h"
#include "acc_runtime.h"
#include "kernels.h"
#include "user_defines_runtime_lib.h"
#include "static_analysis.h"

#include "../acc/string_vec.h"
typedef void (*Kernel)(const int3, const int3, DeviceVertexBufferArray vba);
#define AcReal3(x,y,z)   (AcReal3){x,y,z}
#define AcComplex(x,y)   (AcComplex){x,y}
static bool initialized = false;
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
#include "user_kernel_declarations.h"
#include "kernel_reduce_info.h"



#define USE_COMPRESSIBLE_MEMORY (0)

//TP: unfortunately cannot use color output since it might not be supported in each env
const bool useColor = false;

#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define RESET "\033[0m"

#define COLORIZE(symbol, color) (useColor ? color symbol RESET : symbol)

#include "astaroth_cuda_wrappers.h"
#include "acc/implementation.h"

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
	if(AC_CPU_BUILD) return 1;
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

#include "stencil_accesses.h"
#include "../acc/mem_access_helper_funcs.h"

static Volume
get_bpg(Volume dims, const AcKernel kernel, const int3 block_factors, const Volume tpb)
{
	if(kernel_has_block_loops(kernel)) return get_bpg(ceil_div(dims,block_factors), tpb);
	return get_bpg(dims,tpb);
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
static bool
is_large_launch(const int3 dims)
{
  const int3 ghosts = get_ghosts();
  return ((int)dims.x > ghosts.x && (int)dims.y > ghosts.y && (int)dims.z > ghosts.z);
}

static bool
is_large_launch(const Volume dims)
{
  const int3 ghosts = get_ghosts();
  return ((int)dims.x > ghosts.x && (int)dims.y > ghosts.y && (int)dims.z > ghosts.z);
}


static bool
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
	ERRCHK_CUDA_ALWAYS(
		acOccupancyMaxActiveBlocksPerMultiprocessor(
			&maxBlocksPerSM,
			(void*)kernels[kernel],
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


typedef struct {
  AcKernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;


static TBConfig getOptimalTBConfig(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba);


__device__ __constant__ AcReal* d_symbol_reduce_scratchpads_real[NUM_REAL_SCRATCHPADS];
__device__ __constant__ AcReal  d_reduce_real_res_symbol[NUM_REAL_SCRATCHPADS];


static AcReal* d_reduce_scratchpads_real[NUM_REAL_SCRATCHPADS];
static size_t d_reduce_scratchpads_size_real[NUM_REAL_SCRATCHPADS];
#include "reduce_helpers.h"


void
ac_resize_scratchpads_to_fit(const size_t n_elems, VertexBufferArray vba, const AcKernel kernel)
{
	ac_resize_reals_to_fit(n_elems,vba,kernel);
	ac_resize_ints_to_fit(n_elems,vba,kernel);
#if AC_DOUBLE_PRECISION
	ac_resize_floats_to_fit(n_elems,vba,kernel);
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
#define ac_dummy_write(field,x,y,z) 

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
#if AC_CPU_BUILD

template <typename T>
T
__ldg(const T* src)
{
	return *src;
}
#endif
#include "user_kernels.h"
#undef size
#undef longlong

#include "user_built-in_constants.h"
#include "user_builtin_non_scalar_constants.h"

bool
acRuntimeIsInitialized() { return initialized; }

AcResult
acRuntimeInit(const AcMeshInfo config)
{
  ERRCHK_ALWAYS(initialized == false);
  dimension_inactive = config[AC_dimension_inactive];
  sparse_autotuning  = config[AC_sparse_autotuning];
  raytracing_subblock = config[AC_raytracing_block_factors];
  x_ray_shared_mem_block_size = config[AC_x_ray_shared_mem_block_size];
  z_ray_shared_mem_block_size = config[AC_z_ray_shared_mem_block_size];
  max_tpb_for_reduce_kernels = config[AC_max_tpb_for_reduce_kernels];
  acAllocateArrays(config);
  initialized = true;
  return AC_SUCCESS;
}

AcResult
acLaunchKernelBase(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba, const dim3 bpg, const dim3 tpb, const size_t smem, const cudaStream_t stream)
{
  if(is_coop_raytracing_kernel(kernel) && acSupportsCooperativeLaunches())
  {
	void* args[] = {(void*)&start,(void*)&end,(void*)&vba.on_device};
	ERRCHK_CUDA(acLaunchCooperativeKernel((void*)kernels[kernel],bpg,tpb,args,smem,stream));
  }
  else
  {
  	KERNEL_VBA_LAUNCH(kernels[kernel],bpg,tpb,smem,stream)(start,end,vba.on_device);
  }
  return AC_SUCCESS;
}



static const Volume 
get_kernel_end(const AcKernel kernel, const Volume start, const Volume end)
{
	if(is_raytracing_kernel(kernel) && !AC_CPU_BUILD)
	{
		const auto step_direction = raytracing_step_direction(kernel);
		if(step_direction.z) return (Volume){end.x,end.y,start.z+1};
		if(step_direction.y) return (Volume){end.x,start.y+1,end.z};
		if(step_direction.x) return (Volume){start.x+1,end.y,end.z};
	}
	return (Volume){end.x,end.y,end.z};

}

void
update_reduce_offsets_and_resize(const AcKernel kernel, const int3 start, const dim3 tpb, const dim3 bpg, VertexBufferArray vba)
{
  if (kernel_calls_reduce[kernel] && reduce_offsets[kernel].find(start) == reduce_offsets[kernel].end())
  {
  	reduce_offsets[kernel][start] = kernel_running_reduce_offsets[kernel];
  	kernel_running_reduce_offsets[kernel] += acGetNumOfWarps(bpg,tpb);
	ac_resize_scratchpads_to_fit(kernel_running_reduce_offsets[kernel],vba,kernel);
  }
}


AcResult
acSetReduceOffset(AcKernel kernel, const Volume start_volume,
               const Volume end_volume, VertexBufferArray vba)
{
  const int3 start = to_int3(start_volume);
  const int3 end   = to_int3(get_kernel_end(kernel,start_volume,end_volume));

  const TBConfig tbconf = getOptimalTBConfig(kernel, start, end, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims),kernel,vba.on_device.block_factor, to_volume(tpb)));
  update_reduce_offsets_and_resize(kernel,start,tpb,bpg,vba);
  return AC_SUCCESS;
}

static int3
get_kernel_dims(const AcKernel kernel, const int3 start, const int3 end)
{
  return is_coop_raytracing_kernel(kernel) ? ceil_div(end-start,raytracing_subblock) : end-start;
}

AcResult
acLaunchKernelCommon(AcKernel kernel, const cudaStream_t stream, const int3 start, const int3 end,
	             VertexBufferArray vba, const dim3 tpb)	
{
  const int3 dims       = get_kernel_dims(kernel,start,end);
  const dim3 bpg        = to_dim3(get_bpg(to_volume(dims),kernel,vba.on_device.block_factor, to_volume(tpb)));
  if (kernel_calls_reduce[kernel] && reduce_offsets[kernel].find(start) == reduce_offsets[kernel].end())
  {
	  fprintf(stderr,"Did not find reduce_offset for Kernel launch of %s starting at (%d,%d,%d)!\n",kernel_names[kernel],start.x,start.y,start.z);
	  fflush(stderr);
	  exit(EXIT_FAILURE);
  }
  const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));

  if(kernel_calls_reduce[kernel]) vba.on_device.reduce_offset = reduce_offsets[kernel][start];
  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
  acLaunchKernelBase(kernel,start,end,vba,bpg,tpb,smem,stream);
  ERRCHK_CUDA_KERNEL();

  last_tpb = tpb; // Note: a bit hacky way to get the tpb
  return AC_SUCCESS;
}

AcResult
acLaunchKernelWithTPB(AcKernel kernel, const cudaStream_t stream, const Volume start_volume,
               const Volume end_volume, VertexBufferArray vba, const dim3 tpb)
{
  const int3 start = to_int3(start_volume);
  const int3 end   = to_int3(get_kernel_end(kernel,start_volume,end_volume));
  return acLaunchKernelCommon(kernel,stream,start,end,vba,tpb);
}

AcResult
acLaunchKernel(AcKernel kernel, const cudaStream_t stream, const Volume start_volume,
               const Volume end_volume, VertexBufferArray vba)
{
  const int3 start = to_int3(start_volume);
  const int3 end   = to_int3(get_kernel_end(kernel,start_volume,end_volume));
  const TBConfig tbconf = getOptimalTBConfig(kernel, start, end, vba);
  return acLaunchKernelCommon(kernel,stream,start,end,vba,tbconf.tpb);
}

AcResult
acBenchmarkKernel(AcKernel kernel, const int3 start, const int3 end,
                  VertexBufferArray vba)
{
  const TBConfig tbconf = getOptimalTBConfig(kernel, start, end, vba);
  const dim3 tpb        = tbconf.tpb;
  const int3 dims       = tbconf.dims;
  [[maybe_unused]] const dim3 bpg        = to_dim3(get_bpg(to_volume(dims), to_volume(tpb)));
  [[maybe_unused]] const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER, sizeof(AcReal));

  // Timer create
  cudaEvent_t tstart, tstop;
  ERRCHK_CUDA(acEventCreate(&tstart));
  ERRCHK_CUDA(acEventCreate(&tstop));

  // Warmup
  ERRCHK_CUDA(acEventRecord(tstart));
  KERNEL_LAUNCH(kernels[kernel],bpg, tpb, smem)(start, end, vba.on_device);
  ERRCHK_CUDA(acEventRecord(tstop));
  ERRCHK_CUDA(acEventSynchronize(tstop));
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());

  // Benchmark
  ERRCHK_CUDA(acEventRecord(tstart)); // Timing start
  KERNEL_LAUNCH(kernels[kernel],bpg,tpb,smem)(start, end, vba.on_device);
  ERRCHK_CUDA(acEventRecord(tstop)); // Timing stop
  ERRCHK_CUDA(acEventSynchronize(tstop));
  float milliseconds = 0;
  ERRCHK_CUDA(acEventElapsedTime(&milliseconds, tstart, tstop));

  ERRCHK_ALWAYS(kernel < NUM_KERNELS);
  printf("Kernel %s time elapsed: %g ms\n", kernel_names[kernel],
         static_cast<double>(milliseconds));

  // Timer destroy
  ERRCHK_CUDA(acEventDestroy(tstart));
  ERRCHK_CUDA(acEventDestroy(tstop));

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
  const cudaError_t retval = acMemcpyToSymbol(
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
  const cudaError_t retval = acMemcpyFromSymbol(
      data, stencils, bytes, stencil * bytes, cudaMemcpyDeviceToHost);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};

AcResult
acLoadRealReduceRes(cudaStream_t stream, const AcRealOutputParam param, const AcReal* value)
{
  	const size_t offset =   (size_t)(&d_reduce_real_res_symbol[param]) - (size_t)&d_reduce_real_res_symbol;
	ERRCHK_CUDA(acMemcpyToSymbolAsync(d_reduce_real_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
	return AC_SUCCESS;
}

AcResult
acLoadIntReduceRes(cudaStream_t stream, const AcIntOutputParam param, const int* value)
{
  	const size_t offset =   (size_t)(&d_reduce_int_res_symbol[param]) - (size_t)&d_reduce_int_res_symbol;
	ERRCHK_CUDA(acMemcpyToSymbolAsync(d_reduce_int_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
	return AC_SUCCESS;
}

#if AC_DOUBLE_PRECISION
AcResult
acLoadFloatReduceRes(cudaStream_t stream, const AcFloatOutputParam param, const float* value)
{
  	const size_t offset =   (size_t)&d_reduce_float_res_symbol[param]- (size_t)&d_reduce_float_res_symbol;
	ERRCHK_CUDA(acMemcpyToSymbolAsync(d_reduce_float_res_symbol, value, sizeof(value), offset, cudaMemcpyHostToDevice, stream));
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
  	const cudaError_t retval = acMemcpyToSymbol(&d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice);
  	return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
}

#include "memcpy_to_gmem_arrays.h"
#include "memcpy_from_gmem_arrays.h"


template <typename P, typename V>
AcResult
acStoreUniform(const P param, V* value)
{
	ERRCHK_ALWAYS(param < get_num_params<P>());
	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
  	const size_t offset =  get_address(param) - (size_t)&d_mesh_info;
	const cudaError_t retval = acMemcpyFromSymbol(value, &d_mesh_info, sizeof(V), offset, cudaMemcpyDeviceToHost);
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
		acMemcpyFromGmemArray(array,src_ptr);
		ERRCHK_ALWAYS(src_ptr != nullptr);
		ERRCHK_CUDA_ALWAYS(acMemcpy(values, src_ptr, bytes, cudaMemcpyDeviceToHost));
	}
	else
		ERRCHK_CUDA_ALWAYS(store_array(values, bytes, array));
	return AC_SUCCESS;
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
		acMemcpyFromGmemArray(array,dst_ptr);
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
		ERRCHK_CUDA_ALWAYS(acMemcpy(dst_ptr,values,bytes,cudaMemcpyHostToDevice));
	}
	else 
		ERRCHK_CUDA_ALWAYS(load_array(values, bytes, array));
#if AC_VERBOSE
	fprintf(stderr,"Loaded %s\n",get_name(array));
	fflush(stderr);
#endif
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
    const AcReal percent_of_num_samples = AcReal(num_samples)/AcReal(100.0);
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
make_vtxbuf_input_params_safe(VertexBufferArray& vba, const AcKernel)
{
  //TP: have to set reduce offset zero since it might not be
  vba.on_device.reduce_offset = 0;
//#include "safe_vtxbuf_input_params.h"
}

static AcResult
acResetScratchPadStates(VertexBufferArray vba)
{
  if(vba.scratchpad_states) memset(vba.scratchpad_states,0,sizeof(AcScratchpadStates));
  return AC_SUCCESS;
}


static TBConfig
autotune(const AcKernel kernel, const int3 start, const int3 end, VertexBufferArray vba)
{

  if(AC_CPU_BUILD)
  {
	  return (TBConfig)
	  {
		  kernel,
		  end-start,
	          (dim3){1,1,1}
	  };
  }
  const int3 dims = get_kernel_dims(kernel,start,end);
  make_vtxbuf_input_params_safe(vba,kernel);

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
  if(dims.y == 1 && dims.z == 1)
  {
  	for (int x = x_increment;
  	         x <= min(max_threads_per_block,tpb_end.x); x += x_increment) {
		samples.push_back((int3){x,1,1});
	}

  }
  else if(dims.x == 1 && dims.y == 1)
  {
  	const int z_increment = min(
		  			minimum_transaction_size_in_elems,
					tpb_end.z
		            );
  	for (int z = z_increment;
  	         z <= min(max_threads_per_block,tpb_end.z); z += z_increment) {
  	      samples.push_back((int3){1,1,z});
	}

  }
  else
  {
  	for (int z = 1; z <= min(max_threads_per_block,tpb_end.z); ++z) {
  	  for (int y = 1; y <= min(max_threads_per_block,tpb_end.y); ++y) {
  	    for (int x = x_increment;
  	         x <= min(max_threads_per_block,tpb_end.x); x += x_increment) {


  	      if (x * y * z > max_threads_per_block)
  	        break;
  	      const dim3 tpb{(unsigned int)x, (unsigned int)y, (unsigned int)z};
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
  }
  if(samples.size() == 0)
  {
	fprintf(stderr,"Found no suitable thread blocks for Kernel %s!\n",kernel_names[kernel]);
	fprintf(stderr,"Launch dims (%d:%d,%d:%d,%d:%d)",start.x,end.x,start.y,end.y,start.z,end.z);
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
        const dim3 tpb{(unsigned int)x, (unsigned int)y, (unsigned int)z};
        const dim3 bpg    = to_dim3(
                                get_bpg(to_volume(dims),kernel,vba.on_device.block_factor,
                                to_volume(tpb)
                                ));
	const int n_warps = acGetNumOfWarps(bpg,tpb);
	if(kernel_calls_reduce[kernel])
		ac_resize_scratchpads_to_fit(n_warps,vba,kernel);
        const size_t smem = get_smem(kernel,to_volume(tpb), STENCIL_ORDER,
                                     sizeof(AcReal));

        cudaEvent_t tstart, tstop;
        ERRCHK_CUDA(acEventCreate(&tstart));
        ERRCHK_CUDA(acEventCreate(&tstop));

        acLaunchKernelBase(kernel,start,end,vba,bpg,tpb,smem,0);
        ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
        ERRCHK_CUDA(acEventRecord(tstart)); // Timing start
        for (int i = 0; i < num_iters; ++i)
	{
        	acLaunchKernelBase(kernel,start,end,vba,bpg,tpb,smem,0);
	}
        ERRCHK_CUDA(acEventRecord(tstop)); // Timing stop
        ERRCHK_CUDA(acEventSynchronize(tstop));

        float milliseconds = 0;
        ERRCHK_CUDA(acEventElapsedTime(&milliseconds, tstart, tstop));

        ERRCHK_CUDA(acEventDestroy(tstart));
        ERRCHK_CUDA(acEventDestroy(tstop));
        ++counter;
        if (log) logAutotuningStatus(counter,n_samples,kernel,best_measurement.time / num_iters);

        // Discard failed runs (attempt to clear the error to cudaSuccess)
        const auto err = acGetLastError();
        //TP: it is fine to simply skip invalid configuration values since it can be because of too large tpb's
        //We simply do not count them for finding the optim config
        if(err == cudaErrorInvalidConfiguration) continue;
        if(err == cudaErrorLaunchOutOfResources) continue;
        if (err != cudaSuccess) {
          //TP: reset autotune results
          fprintf(stderr,"\nFailed while autotuning: %s\nReason: %s\n",kernel_names[kernel],acGetErrorName(err));
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
  acResetScratchPadStates(vba);
  return c;
}

static bool
file_exists(const char* filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

int3
acReadOptimTBConfig(const AcKernel kernel, const Volume dims_volume, const Volume block_factors_volume)
{
  const int3 dims = to_int3(dims_volume);
  const int3 block_factors = to_int3(block_factors_volume);
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

  const int3 read_tpb = acReadOptimTBConfig(kernel,to_volume(dims),to_volume(vba.on_device.block_factor));
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
//TP: comes from src/core/kernels/reductions.cc
AcResult
acRuntimeQuit()
{
  	ERRCHK_ALWAYS(initialized == true);
	tbconfigs.clear();
	for(int kernel = 0; kernel < NUM_KERNELS; ++kernel)
	{
		reduce_offsets[kernel].clear();
		kernel_running_reduce_offsets[kernel] = 0;
	}
	initialized = false;
	return AC_SUCCESS;
}
