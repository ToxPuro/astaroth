#define AC_RUNTIME_SOURCE


typedef enum AcBoundary {
    BOUNDARY_NONE  = 0,
    BOUNDARY_X_TOP = 0x01,
    BOUNDARY_X_BOT = 0x02,
    BOUNDARY_X     = BOUNDARY_X_TOP | BOUNDARY_X_BOT,
    BOUNDARY_Y_TOP = 0x04,
    BOUNDARY_Y_BOT = 0x08,
    BOUNDARY_Y     = BOUNDARY_Y_TOP | BOUNDARY_Y_BOT,
    BOUNDARY_Z_TOP = 0x10,
    BOUNDARY_Z_BOT = 0x20,
    BOUNDARY_Z     = BOUNDARY_Z_TOP | BOUNDARY_Z_BOT,
    BOUNDARY_XY    = BOUNDARY_X | BOUNDARY_Y,
    BOUNDARY_XZ    = BOUNDARY_X | BOUNDARY_Z,
    BOUNDARY_YZ    = BOUNDARY_Y | BOUNDARY_Z,
    BOUNDARY_XYZ   = BOUNDARY_X | BOUNDARY_Y | BOUNDARY_Z
} AcBoundary;

bool should_reduce_real[1000] = {false};
bool should_reduce_int[1000] = {false};


#define rocprim__warpSize() (64)
#define rocprim__warpId()   (0)
#define rocprim__warp_shuffle(mask,val)  (val)
#define rocprim__warp_shuffle_down(mask,val)  (val)

#define DEVICE_INLINE
#ifndef AC_IN_AC_LIBRARY
#define AC_IN_AC_LIBRARY
#endif
#define longlong long long
#include "func_attributes.h"

#include <assert.h>

#include <string.h>
#include <vector>
#include "errchk.h"
#include "datatypes.h"

#define AcReal3(x,y,z)   (AcReal3){x,y,z}
#define AcComplex(x,y)   (AcComplex){x,y}
#include "user_defines.h"
#include <array>

#undef __device__
#define __device__
#undef __global__
#define __global__
#undef __launch_bounds__
#define __launch_bounds__(x)
#undef __syncthreads
#define __syncthreads()
#undef __shared__
#define __shared__

#define threadIdx ((int3){0, 0, 0})
#define blockIdx ((dim3){0, 0, 0})
#define blockDim ((dim3){1, 1, 1})
#define gridDim ((dim3){1, 1, 1})
#define make_int3(x, y, z) ((int3){x, y, z})
#define make_float3(x, y, z) ((float3){x, y, z})
#define make_double3(x, y, z) ((double3){x, y, z})
void
print(const char* , ...)
{
}
#define len(arr) sizeof(arr) / sizeof(arr[0])
#define rand_uniform()        (0.5065983774206012) // Chosen by a fair dice roll
#define random_uniform(TID) (0.5065983774206012) // Chosen by a fair dice roll
#define __syncthreads() 

#define vertexIdx ((int3){start.x, start.y, start.z})
#define globalVertexIdx ((int3){vertexIdx.x, vertexIdx.y, vertexIdx.z})

#define localCompdomainVertexIdx (d_mesh_info[AC_nmin])


#define local_compdomain_idx ((LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx))

template <typename T>
T
atomicAdd(T* dst, T val)
{
	*dst += val;
	return val;
}
// Just nasty: Must evaluate all code branches given arbitrary input
// if we want automated stencil generation to work in every case
#define d_multigpu_offset ((int3){0, 0, 0})


constexpr int
IDX(const int i)
{
  (void)i; // Unused
  return 0;
}

int
IDX(const int i, const int j, const int k)
{
  (void)i; // Unused
  (void)j; // Unused
  (void)k; // Unused
  return 0;
}

static int UNUSED
IDX(const int3 idx)
{
  (void)idx; // Unused
  return 0;
}

#define DEVICE_VTXBUF_IDX IDX

static int UNUSED
LOCAL_COMPDOMAIN_IDX(const int3 coord)
{
  (void)coord; // Unused
  return 0;
}
template <typename T>
T
__ldg(T* val)
{
	return *val;
}
#if AC_USE_HIP
uint64_t
__ballot(bool)
{
	return ~0;
}

int
__ffsll(unsigned long long)
{
	return 1;
}

#else
uint64_t
__ballot_sync(unsigned long, bool)
{
	return ~0;
}
template <typename T>
T
__shfl_down_sync(unsigned long, T val, int)
{
	return val;
}
template <typename T>
T
__shfl_sync(unsigned long, T val, int)
{
	return val;
}
int
__ffs(unsigned long)
{
	return 1;
}

#endif
#define idx  ((int)IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z))

#undef  __device__
#define __device__

#undef  __constant__
#define __constant__

#include "math_utils.h"
 
#define constexpr
#include "acc_runtime.h"
#undef constexpr

static int stencils_accessed[NUM_ALL_FIELDS][NUM_STENCILS]{{}};
static int previous_accessed[NUM_ALL_FIELDS+NUM_PROFILES]{};
static int written_fields[NUM_ALL_FIELDS]{};
static int read_fields[NUM_ALL_FIELDS]{};
static int field_has_stencil_op[NUM_ALL_FIELDS]{};
static int read_profiles[NUM_PROFILES]{};
static int reduced_profiles[NUM_PROFILES]{};
static int written_profiles[NUM_PROFILES]{};
static int reduced_reals[NUM_REAL_OUTPUTS+1]{};
static int reduced_ints[NUM_INT_OUTPUTS+1]{};
static int reduced_floats[NUM_FLOAT_OUTPUTS+1]{};

AcKernel current_kernel{};
#define reduce_sum_real_x  reduce_prof
#define reduce_sum_real_y  reduce_prof
#define reduce_sum_real_z  reduce_prof
#define reduce_sum_real_xy reduce_prof
#define reduce_sum_real_xz reduce_prof
#define reduce_sum_real_yx reduce_prof
#define reduce_sum_real_yz reduce_prof
#define reduce_sum_real_zx reduce_prof
#define reduce_sum_real_zy reduce_prof

std::vector<KernelReduceOutput> reduce_outputs{};

void
reduce_sum_real(const bool&, const AcReal, const AcRealOutputParam dst)
{
	if constexpr (NUM_REAL_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No real outputs but reduce_sum_real called!\n");
		fprintf(stderr,"No real outputs but reduce_sum_real called!\n");
		fprintf(stderr,"No real outputs but reduce_sum_real called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_reals[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_reals[dst] = REDUCE_SUM;
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_SUM,current_kernel});
}

void
reduce_max_real(const bool&, const AcReal, const AcRealOutputParam dst)
{
	if constexpr (NUM_REAL_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No real outputs but reduce_max_real called!\n");
		fprintf(stderr,"No real outputs but reduce_max_real called!\n");
		fprintf(stderr,"No real outputs but reduce_max_real called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_reals[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_reals[dst] = REDUCE_MAX;
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MAX,current_kernel});
}

void
reduce_min_real(const bool&, const AcReal, const AcRealOutputParam dst)
{
	if constexpr (NUM_REAL_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No real outputs but reduce_min_real called!\n");
		fprintf(stderr,"No real outputs but reduce_min_real called!\n");
		fprintf(stderr,"No real outputs but reduce_min_real called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_reals[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",real_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_reals[dst] = REDUCE_MIN;
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MIN,current_kernel});
}

void
reduce_sum_int(const bool&, const int, const AcIntOutputParam dst)
{
	if constexpr (NUM_INT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No int outputs but reduce_sum_int called!\n");
		fprintf(stderr,"No int outputs but reduce_sum_int called!\n");
		fprintf(stderr,"No int outputs but reduce_sum_int called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_ints[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_ints[dst] = REDUCE_SUM;
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_SUM,current_kernel});
}

void
reduce_max_int(const bool&, const int, const AcIntOutputParam dst)
{
	if constexpr (NUM_INT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No int outputs but reduce_max_int called!\n");
		fprintf(stderr,"No int outputs but reduce_max_int called!\n");
		fprintf(stderr,"No int outputs but reduce_max_int called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_ints[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_ints[dst] = REDUCE_MAX;
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MAX,current_kernel});
}

void
reduce_min_int(const bool&, const int, const AcIntOutputParam dst)
{
	if constexpr (NUM_INT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No int outputs but reduce_min_int called!\n");
		fprintf(stderr,"No int outputs but reduce_min_int called!\n");
		fprintf(stderr,"No int outputs but reduce_min_int called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_ints[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",int_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_ints[dst] = REDUCE_MIN;
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MIN,current_kernel});
}

void
reduce_sum_float(const bool&, const float, const AcFloatOutputParam dst)
{
	if constexpr (NUM_FLOAT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No float outputs but reduce_sum_float called!\n");
		fprintf(stderr,"No float outputs but reduce_sum_float called!\n");
		fprintf(stderr,"No float outputs but reduce_sum_float called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_floats[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_floats[dst] = REDUCE_SUM;
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_SUM,current_kernel});
}

void
reduce_max_float(const bool&, const float, const AcFloatOutputParam dst)
{
	if constexpr (NUM_FLOAT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No float outputs but reduce_max_float called!\n");
		fprintf(stderr,"No float outputs but reduce_max_float called!\n");
		fprintf(stderr,"No float outputs but reduce_max_float called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_floats[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_floats[dst] = REDUCE_MAX;
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_MAX,current_kernel});
}

void
reduce_min_float(const bool&, const float, const AcFloatOutputParam dst)
{
	if constexpr (NUM_FLOAT_OUTPUTS == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No float outputs but reduce_min_float called!\n");
		fprintf(stderr,"No float outputs but reduce_min_float called!\n");
		fprintf(stderr,"No float outputs but reduce_min_float called!\n");
		exit(EXIT_FAILURE);
	}
	if(reduced_floats[dst]) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		fprintf(stderr,"Can not reduce %s more than once in %s!\n",float_output_names[dst],kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
	reduced_floats[dst] = REDUCE_MIN;
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_MIN,current_kernel});
}


void
reduce_prof(const bool&, const AcReal, const Profile dst)
{
	if constexpr (NUM_PROFILES == 0) 
	{
		fprintf(stderr,"\nFATAL AC ERROR:\n");
		fprintf(stderr,"No profiles but trying to do a profile reduction!\n");
		fprintf(stderr,"No profiles but trying to do a profile reduction!\n");
		fprintf(stderr,"No profiles but trying to do a profile reduction!\n");
		exit(EXIT_FAILURE);
	}
	reduced_profiles[(int)dst] = REDUCE_SUM;
	reduce_outputs.push_back({(int)dst,AC_PROF_TYPE, REDUCE_SUM,current_kernel});
}


#define value_profile_x value_profile
#define value_profile_y value_profile
#define value_profile_z value_profile
#define value_profile_xy value_profile
#define value_profile_xz value_profile
#define value_profile_yx value_profile
#define value_profile_yz value_profile
#define value_profile_zx value_profile
#define value_profile_zy value_profile
AcReal
value_profile(const Profile& prof)
{
	if constexpr (NUM_PROFILES != 0)
		read_profiles[prof] = 1;
	return 0.0;
}

static UNUSED const char* 
get_name(const AcRealOutputParam& param)
{
        return real_output_names[param];
}
static UNUSED const char* 
get_name(const AcIntOutputParam& param)
{
        return int_output_names[param];
}

static UNUSED const char* 
get_name(const Profile& param)
{
	return profile_names[param];
}

 

extern "C" 
{
	AcResult acAnalysisGetKernelInfo(const AcMeshInfoParams info, KernelAnalysisInfo* src);
	acAnalysisBCInfo acAnalysisGetBCInfo(const AcMeshInfoParams info, const AcKernel bc, const AcBoundary boundary);
}
//#include "user_constants.h"
typedef void (*Kernel)(const int3, const int3, DeviceVertexBufferArray vba);
#define tid  ((int3){0,0,0})
#include "user_kernel_declarations.h"

constexpr AcMeshInfoScalars
get_d_mesh_info()
{
	//TP: DCONST ints and bools have to be evaluated to 1 since PC depends on conditionals like if(int) and if(bool) being true at analysis 
	AcMeshInfoParams res{};
  	for(int i = 0; i < NUM_INT_PARAMS; ++i)
	  res.scalars.int_params[i] = 1;
  	for(int i = 0; i < NUM_BOOL_PARAMS; ++i)
	  res.scalars.bool_params[i] = true;
	return res.scalars;
}

static AcMeshInfoScalars  d_mesh_info = get_d_mesh_info();

AcResult
acAnalysisLoadMeshInfo(const AcMeshInfoParams info) 
{d_mesh_info = info.scalars; return AC_SUCCESS;}

#include "dconst_decl.h"
#include "rconst_decl.h"

#include "dconst_arrays_decl.h"
#include "gmem_arrays_accessed_decl.h"
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME) static DATATYPE ARR_NAME##return_var{}; \
							    struct tmp_struct_##ARR_NAME {DATATYPE& operator[](const int) {gmem_##DEFINE_NAME##_arrays_accessed[ARR_NAME] = 1; return ARR_NAME##return_var;}}; \
							    [[maybe_unused]] static tmp_struct_##ARR_NAME AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME {};

#define DECLARE_CONST_DIMS_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, DIMS) static DATATYPE ARR_NAME##return_var{}; \
							    struct tmp_struct_##ARR_NAME {DATATYPE& operator[](const int) {gmem_##DEFINE_NAME##_arrays_accessed[ARR_NAME] = 1; return ARR_NAME##return_var;}}; \
							    [[maybe_unused]] static tmp_struct_##ARR_NAME AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME {};
#include "gmem_arrays_decl.h"

AcReal smem[8 * 1024 * 1024]; // NOTE: arbitrary limit: need to allocate at
                              // least the max smem size of the device
[[maybe_unused]] constexpr int AC_IN_BOUNDS_WRITE      = (1 << 0);
[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_WRITE  = (1 << 1);

[[maybe_unused]] constexpr int AC_IN_BOUNDS_READ      = (1 << 0);
[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_READ  = (1 << 1);
[[maybe_unused]] constexpr int AC_STENCIL_CALL        = (1 << 2);
#include "analysis_stencils.h"


int3
VAL(const AcInt3CompParam&)
{
	return (int3){0,0,0};
}
bool
index_at_boundary(const int x, const int y, const int z)
{
#include "user_builtin_non_scalar_constants.h"
	return  
	      ((x < VAL(AC_nmin).x) || (x >= VAL(AC_nlocal_max).x))
	   || ((y < VAL(AC_nmin).y) || (y >= VAL(AC_nlocal_max).y))
	   || ((z < VAL(AC_nmin).z) || (z >= VAL(AC_nlocal_max).z))
	   ;
}
void
mark_as_written(const Field& field, const int x, const int y, const int z)
{
	written_fields[field] |= 
			index_at_boundary(x,y,z) ? AC_IN_BOUNDS_WRITE : AC_OUT_OF_BOUNDS_WRITE;
}
void
write_base (const Field& field, const AcReal&)
{
	written_fields[field] |= AC_IN_BOUNDS_WRITE;
}
void
write_to_index (const Field& field, const int&, const AcReal&)
{
	written_fields[field] |= AC_OUT_OF_BOUNDS_WRITE;
}
AcReal
previous_base(const Field& field)
{
	previous_accessed[field] = 1;
	return AcReal(1.0);
}
AcReal
AC_INTERNAL_read_field(const Field& field, const int x, const int y, const int z)
{
	stencils_accessed[field][stencil_value_stencil] |= 
							index_at_boundary(x,y,z) ? AC_IN_BOUNDS_READ : AC_OUT_OF_BOUNDS_READ;
	return AcReal(1.0);
}
#define suppress_unused_warning(X) (void)X

static std::vector<int> executed_nodes{};
#define constexpr
#define size(arr) (int)(sizeof(arr)/sizeof(arr[0]))
#define min(a,b) a < b ? a : b
#include "user_cpu_kernels.h"
#undef  constexpr
#undef size



VertexBufferArray
vbaCreate(const size_t count)
{
  VertexBufferArray vba{};
  memset(&vba, 0, sizeof(vba));

  const size_t bytes = sizeof(vba.on_device.in[0][0]) * count;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.on_device.in[i]  = (AcReal*)malloc(bytes);
    vba.on_device.out[i] = (AcReal*)malloc(bytes);
  }
  for (size_t i = 0; i < NUM_PROFILES; ++i) {
    vba.on_device.profiles.in[i]  = (AcReal*)malloc(bytes);
    vba.on_device.profiles.out[i] = (AcReal*)malloc(bytes);
  }

  vba.on_device.block_factor = (int3){1,1,1};
  return vba;
}

void
vbaDestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    free(vba->on_device.in[i]);
    free(vba->on_device.out[i]);
    vba->on_device.in[i]  = NULL;
    vba->on_device.out[i] = NULL;
  }
  for (size_t i = 0; i < NUM_PROFILES; ++i) {
    free(vba->on_device.profiles.in[i]);
    free(vba->on_device.profiles.out[i]);
    vba->on_device.profiles.in[i]  = NULL;
    vba->on_device.profiles.out[i] = NULL;
  }
}
VertexBufferArray VBA = vbaCreate(1000);

void
execute_kernel(const int kernel_index)
{
    const Kernel kernel = kernels[kernel_index];
    current_kernel = (AcKernel)kernel_index;
    kernel((int3){0, 0, 0}, (int3){1, 1, 1}, VBA.on_device);
}
void
execute_kernel(const AcKernel kernel_index, const AcBoundary boundary)
{
#include "user_builtin_non_scalar_constants.h"

        current_kernel = (AcKernel)kernel_index;
    	const Kernel kernel = kernels[kernel_index];
	if(BOUNDARY_X_BOT & boundary)
	{
		int3 start = (int3){0,NGHOST,NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}

	if(BOUNDARY_X_TOP & boundary)
	{
		int3 start = (int3){VAL(AC_nlocal).x+NGHOST, NGHOST, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Y_BOT & boundary)
	{
		int3 start = (int3){NGHOST, 0, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Y_TOP & boundary)
	{
		int3 start = (int3){NGHOST, VAL(AC_nlocal).y+NGHOST, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Z_BOT  & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, 0};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Z_TOP & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, VAL(AC_nlocal).z+NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
}
int
get_executed_nodes()
{ 
  executed_nodes= {};
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
	  execute_kernel(k);
  }
  FILE* fp_executed_nodes= fopen("executed_nodes.bin", "wb");
  const int size = executed_nodes.size();
  fwrite(&size, sizeof(int), 1, fp_executed_nodes);
  fwrite(executed_nodes.data(), sizeof(int), executed_nodes.size(), fp_executed_nodes);
  fclose(fp_executed_nodes);
  return EXIT_SUCCESS;
}	

void
reset_info_arrays()
{
    memset(stencils_accessed, 0,sizeof(stencils_accessed));
    memset(read_fields,0, sizeof(read_fields));
    memset(field_has_stencil_op,0, sizeof(field_has_stencil_op));
    memset(written_fields, 0,    sizeof(written_fields));
    memset(previous_accessed, 0, sizeof(previous_accessed));
    std::vector<KernelReduceOutput> empty_vec{};
    reduce_outputs = empty_vec;
    //TP: would use memset but at least on Puhti the C++ compiler gives an incorrect warning that I am not multiplying the element size even though I am (I guess because the compiler simply sees zero if there are no profiles?)
    for(int i = 0; i  < NUM_PROFILES; ++i)
    {
	    read_profiles[i] = 0;
	    reduced_profiles[i] = 0;
	    written_profiles[i] = 0;
    }
    for(int i = 0; i < NUM_REAL_OUTPUTS; ++i)
	    reduced_reals[i] = 0;
    for(int i = 0; i < NUM_INT_OUTPUTS; ++i)
	    reduced_ints[i] = 0;
    for(int i = 0; i < NUM_FLOAT_OUTPUTS; ++i)
	    reduced_floats[i] = 0;
}

acAnalysisBCInfo 
acAnalysisGetBCInfo(const AcMeshInfoParams info, const AcKernel bc, const AcBoundary boundary)
{
	bool larger_input  = false;
	bool larger_output = false;
	d_mesh_info = info.scalars;
	reset_info_arrays();
    	execute_kernel(bc,boundary);
    	for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
    	{
    	  for (size_t i = 0; i < NUM_STENCILS; ++i)
    	  {
    	    if (stencils_accessed[j][i])
    	    {
    	      if(i == 0) read_fields[j] |= stencils_accessed[j][i];
    	      field_has_stencil_op[j] |= (i != 0);
    	    }
    	    read_fields[j] |= previous_accessed[j];
    	  }
    	}

	for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
	{
		larger_input  |= (read_fields[i]    & AC_IN_BOUNDS_READ);
		larger_output |= (written_fields[i] & AC_OUT_OF_BOUNDS_WRITE);
	}
	return (acAnalysisBCInfo){larger_input,larger_output};
}
AcResult
acAnalysisGetKernelInfo(const AcMeshInfoParams info, KernelAnalysisInfo* dst)
{
	d_mesh_info = info.scalars;
	memset(dst->stencils_accessed,false,sizeof(dst->stencils_accessed));
	for(size_t k = 0; k <NUM_KERNELS; ++k)
	{
		reset_info_arrays();
    		if (!skip_kernel_in_analysis[k])
    		{
    			execute_kernel(k);
    			for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
    			{
    			  for (size_t i = 0; i < NUM_STENCILS; ++i)
    			  {
    			    if (stencils_accessed[j][i])
    			    {
    		              if(i == 0) read_fields[j] |= stencils_accessed[j][i];
    			      field_has_stencil_op[j] |= (i != 0);
    			    }
			    dst->stencils_accessed[k][j][i] |= stencils_accessed[j][i];
    			    read_fields[j] |= previous_accessed[j];
    			  }
    			}
    		}
		for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
		{
			dst->read_fields[k][i]    = read_fields[i];
			dst->field_has_stencil_op[k][i] = field_has_stencil_op[i];
			dst->written_fields[k][i] = written_fields[i];
		}
		for(size_t i = 0; i < NUM_PROFILES; ++i)
		{
			dst->read_profiles[k][i]    = read_profiles[i];
			dst->reduced_profiles[k][i] = reduced_profiles[i];
			dst->written_profiles[k][i] = written_profiles[i];
		}
		dst->n_reduce_outputs[k] = reduce_outputs.size();
		for(size_t i = 0; i < reduce_outputs.size(); ++i)
			dst->reduce_outputs[k][i] = reduce_outputs[i];
		if(dst->n_reduce_outputs[k] > NUM_OUTPUTS)
		{
			fprintf(stderr,"Can not reduce variables multiple times in a Kernel\n");
			exit(EXIT_FAILURE);
		}
	}
	return AC_SUCCESS;
}
template <const size_t N>
void
print_info_array(FILE* fp, const char* name, const int arr[NUM_KERNELS][N])
{
  fprintf(fp,
          "static int %s[NUM_KERNELS][%ld] "
          "__attribute__((unused)) =  {",name,N);
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    fprintf(fp,"{");
    for (size_t j = 0; j < N; ++j)
        fprintf(fp, "%d,", arr[k][j]);
    fprintf(fp,"},");
  }
  fprintf(fp, "};");
}

void
print_info_array(FILE* fp, const char* name, const int arr[NUM_KERNELS][0])
{
  (void)arr;
  fprintf(fp,
          "static int %s[NUM_KERNELS][%d] "
          "__attribute__((unused)) =  {",name,0);
  fprintf(fp, "};");
}

#if AC_STENCIL_ACCESSES_MAIN
int
main(int argc, char* argv[])
{
  //TP: Some Pencil Code code at the moment depends on the unsafe fact that all dconst ints are evaluated as 1 during analysis
  //Will remove this comment when Pencil Code does not depend on this fact anymore
  if (argc != 2) {
    fprintf(stderr, "Usage: ./main <output_file>\n");
    return EXIT_FAILURE;
  }
  if(!strcmp(argv[1],"-C")) return get_executed_nodes();
  const char* output = argv[1];
  FILE* fp           = fopen(output, "w+");
  assert(fp);
  FILE* fp_fields_read = fopen("user_read_fields.bin","wb");
  FILE* fp_written_fields = fopen("user_written_fields.bin", "wb");
  FILE* fp_field_has_stencil_op    = fopen("user_field_has_stencil_op.bin","wb");
  FILE* fp_field_has_previous_call = fopen("user_field_has_previous_call.bin","wb");
  FILE* fp_profiles_read = fopen("user_read_profiles.bin","wb");
  FILE* fp_profiles_reduced = fopen("user_reduced_profiles.bin","wb");
  FILE* fp_reals_reduced = fopen("user_reduced_reals.bin","wb");
  FILE* fp_ints_reduced = fopen("user_reduced_ints.bin","wb");
  FILE* fp_floats_reduced = fopen("user_reduced_floats.bin","wb");

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS] "
          "__attribute__((unused)) =  {");
  int  write_output[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES]{};
  int  output_previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES]{};
  int  output_reduced_profiles[NUM_KERNELS][NUM_PROFILES]{};
  int  output_reduced_reals[NUM_KERNELS][NUM_REAL_OUTPUTS+1]{};
  int  output_reduced_ints[NUM_KERNELS][NUM_INT_OUTPUTS+1]{};
  int  output_reduced_floats[NUM_KERNELS][NUM_FLOAT_OUTPUTS+1]{};
  int  output_read_profiles[NUM_KERNELS][NUM_PROFILES]{};
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    reset_info_arrays();
    fprintf(fp,"{");
    if (!skip_kernel_in_analysis[k])
    {
    	execute_kernel(k);
    	for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
    	{ 
    	  for (size_t i = 0; i < NUM_STENCILS; ++i)
    	  {
    	    if (stencils_accessed[j][i])
    	    {
	      if(i == 0) read_fields[j] |= stencils_accessed[j][i];
    	      field_has_stencil_op[j] |= (i != 0);
    	    }
    	    read_fields[j] |= previous_accessed[j];
    	  }
    	  output_previous_accessed[k][j] = previous_accessed[j];
	  write_output[k][j] = written_fields[j];
    	}
	for(size_t j = 0; j < NUM_PROFILES; ++j)
	{
	  output_reduced_profiles[k][j] = reduced_profiles[j];
	  output_read_profiles[k][j]    = read_profiles[j];
	}
	for(int j = 0; j < NUM_REAL_OUTPUTS; ++j)
		output_reduced_reals[k][j] = reduced_reals[j];
	for(int j = 0; j < NUM_INT_OUTPUTS; ++j)
		output_reduced_ints[k][j] = reduced_ints[j];
	for(int j = 0; j < NUM_FLOAT_OUTPUTS; ++j)
		output_reduced_floats[k][j] = reduced_floats[j];
    } 

    for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
    { 
      fprintf(fp,"{");
      for (size_t i = 0; i < NUM_STENCILS; ++i)
      {
        fprintf(fp,"%d,",stencils_accessed[j][i]);
      }
      fprintf(fp,"},");
    }
    fprintf(fp,"},");

    fwrite(read_fields,sizeof(int), NUM_ALL_FIELDS,fp_fields_read);
    fwrite(field_has_stencil_op,sizeof(int), NUM_ALL_FIELDS,fp_field_has_stencil_op);
    fwrite(previous_accessed,sizeof(int), NUM_ALL_FIELDS,fp_field_has_previous_call);
    fwrite(written_fields,sizeof(int),NUM_ALL_FIELDS,fp_written_fields);
    fwrite(read_profiles   ,sizeof(int),NUM_PROFILES,fp_profiles_read);
    fwrite(reduced_profiles,sizeof(int),NUM_PROFILES,fp_profiles_reduced);
    fwrite(reduced_reals,sizeof(int),NUM_REAL_OUTPUTS,fp_reals_reduced);
    fwrite(reduced_ints,sizeof(int),NUM_INT_OUTPUTS,fp_ints_reduced);
    fwrite(reduced_floats,sizeof(int),NUM_FLOAT_OUTPUTS,fp_floats_reduced);
  }


  fprintf(fp, "};");

  fclose(fp_written_fields);
  fclose(fp_fields_read);
  fclose(fp_field_has_stencil_op);
  fclose(fp_profiles_read);
  fclose(fp_profiles_reduced);

  print_info_array(fp,"previous_accessed",output_previous_accessed);
  print_info_array(fp,"write_called",write_output);
  print_info_array(fp,"reduced_profiles",output_reduced_profiles);
  print_info_array(fp,"read_profiles",output_read_profiles);
  print_info_array(fp,"reduced_reals",output_reduced_reals);
  print_info_array(fp,"reduced_ints",output_reduced_ints);
  print_info_array(fp,"reduced_floats",output_reduced_floats);
  fprintf(fp,"const bool has_mem_access_info __attribute__((unused)) = true;\n");

  fclose(fp);


#include "gmem_arrays_output_accesses.h"
  fprintf(stderr,"Generated stencil accesses\n");
  return EXIT_SUCCESS;
}
#endif
