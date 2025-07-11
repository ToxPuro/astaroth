#define AC_RUNTIME_SOURCE



bool should_reduce_real[1000] = {false};
bool should_reduce_int[1000] = {false};



#define rocprim__warpSize() (64)
#define rocprim__warpId()   (0)
#define rocprim__warp_shuffle(mask,val)  (val)
#define rocprim__warp_shuffle_down(val,offset)  (val)

#include <algorithm>

#ifndef AC_IN_AC_LIBRARY
#define AC_IN_AC_LIBRARY
#endif
#define longlong long long
#include "func_attributes.h"
#include <assert.h>

#include <string.h>
#include <vector>
#include "device_headers.h"
#include "datatypes.h"

#define AcReal3(x,y,z)   (AcReal3){x,y,z}
#define AcComplex(x,y)   (AcComplex){x,y}
#include "user_defines.h"
#include <array>
AcReal AC_INTERNAL_run_const_AcReal_array_here[2000]{};
AcReal AC_INTERNAL_run_const_array_here[2000]{};
bool   AC_INTERNAL_run_const_bool_array_here[2000]{};
int    AC_INTERNAL_run_const_int_array_here[2000]{};
float  AC_INTERNAL_run_const_float_array_here[2000]{};
AcComplex    AC_INTERNAL_run_const_AcComplex_array_here[2000]{};

AcReal*
RCONST(AcRealCompArrayParam)
{
       return AC_INTERNAL_run_const_AcReal_array_here;
}

bool*
RCONST(AcBoolCompArrayParam)
{
       return AC_INTERNAL_run_const_bool_array_here;
}


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

#include "math_utils.h"
 
#define constexpr
#include "acc_runtime.h"
#undef constexpr

static int stencils_accessed[NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS]{{}};
static int previous_accessed[NUM_ALL_FIELDS+NUM_PROFILES]{};
static int incoming_ray_value_accessed[NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1]{};
static int outgoing_ray_value_accessed[NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1]{};
static int written_fields[NUM_ALL_FIELDS]{};
static int written_complex_fields[NUM_COMPLEX_FIELDS+1]{};
static int read_complex_fields[NUM_COMPLEX_FIELDS+1]{};
static int read_fields[NUM_ALL_FIELDS]{};
static int field_has_stencil_op[NUM_ALL_FIELDS]{};
static int read_profiles[NUM_PROFILES]{};
static int reduced_profiles[NUM_PROFILES]{};
static int written_profiles[NUM_PROFILES]{};
static int reduced_reals[NUM_REAL_OUTPUTS+1]{};
static int reduced_ints[NUM_INT_OUTPUTS+1]{};
#if AC_DOUBLE_PRECISION
static int reduced_floats[NUM_FLOAT_OUTPUTS+1]{};
#endif

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

std::vector<KernelReduceOutput> reduce_inputs{};
std::vector<KernelReduceOutput> reduce_outputs{};

AcReal
output_value(const AcRealOutputParam& param)
{
	reduce_inputs.push_back((KernelReduceOutput){(int)param,AC_REAL_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
	return (AcReal){};
}

int
output_value(const AcIntOutputParam& param)
{
	reduce_inputs.push_back((KernelReduceOutput){(int)param,AC_REAL_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
	return (int){};
}

#if AC_DOUBLE_PRECISION
float
output_value(const AcFloatOutputParam& param)
{
	reduce_inputs.push_back((KernelReduceOutput){(int)param,AC_FLOAT_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
	return (float){};
}
#endif

void
postprocess_reduce_result(const AcRealOutputParam dst, const AcReductionPostProcessingOp op)
{
	bool found = false;
	for(auto& output : reduce_outputs)
	{
		if(output.variable == dst && output.type == AC_REAL_TYPE)
		{
			found = true;
			output.postprocess_op = op;
		}
	}
	if(!found)
	{
		fprintf(stderr,"Applied postprocessing op on %s, but it is not reduced in %s!\n",real_output_names[dst], kernel_names[current_kernel]);
		fprintf(stderr,"Applied postprocessing op on %s, but it is not reduced in %s!\n",real_output_names[dst], kernel_names[current_kernel]);
		fprintf(stderr,"Applied postprocessing op on %s, but it is not reduced in %s!\n",real_output_names[dst], kernel_names[current_kernel]);
		exit(EXIT_FAILURE);
	}
}
void
reduce_sum_real(const AcReal, const AcRealOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_max_real(const AcReal, const AcRealOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MAX,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_min_real(const AcReal, const AcRealOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MIN,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_sum_int(const int, const AcIntOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_max_int(const int, const AcIntOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MAX,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_min_int(const int, const AcIntOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MIN,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

#if AC_DOUBLE_PRECISION
void
reduce_sum_float(const float, const AcFloatOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_max_float(const float, const AcFloatOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_MAX,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}

void
reduce_min_float(const float, const AcFloatOutputParam dst)
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
	reduce_outputs.push_back({(int)dst,AC_FLOAT_TYPE,REDUCE_MIN,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
}
#endif


void
reduce_prof(const AcReal, const Profile dst)
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
	reduce_outputs.push_back({(int)dst,AC_PROF_TYPE, REDUCE_SUM,AC_NO_REDUCE_POST_PROCESSING,current_kernel});
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
	void acAnalysisCheckForDSLErrors(const AcMeshInfo info);
	AcResult acAnalysisGetKernelInfo(const AcMeshInfo info, KernelAnalysisInfo* src);
	KernelAnalysisInfo acAnalysisGetKernelInfoSingle(const AcMeshInfo info, const AcKernel kernel);
	acAnalysisBCInfo acAnalysisGetBCInfo(const AcMeshInfo info, const AcKernel bc, const AcBoundary boundary);
}
//#include "user_constants.h"
typedef void (*Kernel)(const int3, const int3, DeviceVertexBufferArray vba);
#define tid  ((int3){0,0,0})
#include "user_kernel_declarations.h"

constexpr AcMeshInfo
get_d_mesh_info()
{
	//TP: DCONST ints and bools have to be evaluated to 1 since PC depends on conditionals like if(int) and if(bool) being true at analysis 
	AcMeshInfo res{};
  	for(int i = 0; i < NUM_INT_PARAMS; ++i)
	  res.int_params[i] = 1;
  	for(int i = 0; i < NUM_INT3_PARAMS; ++i)
	  res.int3_params[i] = (int3){1,1,1};
  	for(int i = 0; i < NUM_BOOL_PARAMS; ++i)
	  res.bool_params[i] = true;
	return res;
}

static AcMeshInfo  d_mesh_info = get_d_mesh_info();

AcResult
acAnalysisLoadMeshInfo(const AcMeshInfo info) 
{d_mesh_info = info ; return AC_SUCCESS;}

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
#include "cpu_gmem_arrays_decl.h"

AcReal smem[8 * 1024 * 1024]; // NOTE: arbitrary limit: need to allocate at
                              // least the max smem size of the device
[[maybe_unused]] constexpr int AC_IN_BOUNDS_WRITE      = (1 << 0);
[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_WRITE  = (1 << 1);
[[maybe_unused]] constexpr int AC_WRITE_TO_INPUT  = (1 << 2);

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
index_at_boundary(const int x, const int y, const int z);
void
mark_as_written(const Field& field, const int x, const int y, const int z)
{
	written_fields[field] |= 
			index_at_boundary(x,y,z) ? AC_IN_BOUNDS_WRITE : AC_OUT_OF_BOUNDS_WRITE;
}
void
mark_as_written(const Field3& field, const int x, const int y, const int z)
{
	mark_as_written(field.x,x,y,z);
	mark_as_written(field.y,x,y,z);
	mark_as_written(field.z,x,y,z);
}

void
ac_dummy_write(const Field& field, const int x, const int y, const int z)
{
	mark_as_written(field, x,y,z);
	written_fields[field] |= AC_WRITE_TO_INPUT;
}


void
AC_INTERNAL_write_vtxbuf(const Field& field, const int x, const int y, const int z, const AcReal&)
{
	mark_as_written(field, x,y,z);
	written_fields[field] |= AC_WRITE_TO_INPUT;
}

void
AC_INTERNAL_write_vtxbuf_at_current_point(const Field& field, const AcReal& val)
{
	AC_INTERNAL_write_vtxbuf(field,0,0,0,val);
}

void
AC_INTERNAL_write_vtxbuf3(const Field3& field, const int x, const int y, const int z, const AcReal3& val)
{
	AC_INTERNAL_write_vtxbuf(field.x,x,y,z,val.x);
	AC_INTERNAL_write_vtxbuf(field.y,x,y,z,val.y);
	AC_INTERNAL_write_vtxbuf(field.z,x,y,z,val.z);
}

void
AC_INTERNAL_write_vtxbuf4(const Field4& field, const int x, const int y, const int z, const AcReal4& val)
{
	AC_INTERNAL_write_vtxbuf(field.x,x,y,z,val.x);
	AC_INTERNAL_write_vtxbuf(field.y,x,y,z,val.y);
	AC_INTERNAL_write_vtxbuf(field.z,x,y,z,val.z);
	AC_INTERNAL_write_vtxbuf(field.w,x,y,z,val.w);
}

static int3 UNUSED
ac_get_field_halos(const Field& field)
{
	return VAL(vtxbuf_halos[field]);
}
void
write_base (const Field& field, const AcReal&)
{
	written_fields[field] |= AC_IN_BOUNDS_WRITE;
}

void
write_complex_base (const ComplexField& field, const AcComplex&)
{
	written_complex_fields[field] |= AC_IN_BOUNDS_WRITE;
}
AcComplex
value_complex(const ComplexField& field)
{
	read_complex_fields[field] |= 1;
	return (AcComplex){0.0,0.0};
}

void
write_at_point (const Field& field, const AcReal&, const int x, const int y, const int z)
{
	mark_as_written(field,x,y,z);
}
template <typename T, typename T2>
AcReal
safe_access(T arr, const int, const int index, const T2)
{
	//TP: not sure do the analysis indeces always correspond that well to the actual indeces so skip this for now
	//if(index < 0 || index >= dims)
	//{
	//	fprintf(stderr,"Trying to access %s out of bounds!: %d\n",real_array_names[param],index);
	//	exit(EXIT_FAILURE);
	//}
	return arr[index];
}

#define write_profile_x  write_profile
#define write_profile_y  write_profile
#define write_profile_z  write_profile
#define write_profile_xy write_profile
#define write_profile_xz write_profile
#define write_profile_yx write_profile
#define write_profile_yz write_profile
#define write_profile_zx write_profile
#define write_profile_zy write_profile
AcReal
write_profile(const Profile& prof, const AcReal&)
{
	if constexpr (NUM_PROFILES != 0)
		written_profiles[prof] = 1;
	return 0.0;
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
AC_INTERNAL_read_vtxbuf(const Field& field, const int x, const int y, const int z)
{
	//TP: this is possible in case of input fields for kernels and when array syntax is translated to a call of this
	if(field >= NUM_FIELDS) return AcReal(1.0);
	stencils_accessed[field][stencil_value_stencil] |= 
							index_at_boundary(x,y,z) ? AC_IN_BOUNDS_READ : AC_OUT_OF_BOUNDS_READ;
	return AcReal(1.0);
}

AcReal3
AC_INTERNAL_read_vtxbuf3(const Field3& field, const int x, const int y, const int z)
{
	return (AcReal3)
	{
		AC_INTERNAL_read_vtxbuf(field.x,x,y,z),
		AC_INTERNAL_read_vtxbuf(field.y,x,y,z),
		AC_INTERNAL_read_vtxbuf(field.z,x,y,z),
	};
}

AcReal4
AC_INTERNAL_read_vtxbuf4(const Field4& field, const int x, const int y, const int z)
{
	return (AcReal4)
	{
		AC_INTERNAL_read_vtxbuf(field.x,x,y,z),
		AC_INTERNAL_read_vtxbuf(field.y,x,y,z),
		AC_INTERNAL_read_vtxbuf(field.z,x,y,z),
		AC_INTERNAL_read_vtxbuf(field.w,x,y,z),
	};
}
AcReal
AC_INTERNAL_read_profile(const Profile& profile, const int)
{
	if constexpr(NUM_PROFILES > 0)
		read_profiles[profile] |= 1;
	return AcReal(1.0);
}
AcReal
AC_INTERNAL_read_profile(const Profile& profile, const int, const int)
{
	if constexpr(NUM_PROFILES > 0)
		read_profiles[profile] |= 1;
	return AcReal(1.0);
}
AcReal3
AC_INTERNAL_read_profile(const VecZProfile& profile, const int)
{
	if constexpr(NUM_PROFILES > 0)
	{
		read_profiles[profile.x] |= 1;
		read_profiles[profile.y] |= 1;
		read_profiles[profile.z] |= 1;
	}
	return (AcReal3){1.0,1.0,1.0};
}
#define suppress_unused_warning(X) (void)X

static std::vector<int> executed_nodes{};
#define constexpr
#define size(arr) (int)(sizeof(arr)/sizeof(arr[0]))

template <typename T1,typename T2>
AcReal3
matmul_arr(T1, T2)
{
	return (AcReal3)
	{
			(AcReal)0.0,
			(AcReal)0.0,
			(AcReal)0.0
	};
}
static bool check_for_errors = false;
void
error_message(const bool error, const char* message)
{
	if(error && check_for_errors)
	{
		fprintf(stderr,"\nAstaroth DSL error: %s\n\n",message);
	}
}

void
fatal_error_message(const bool error, const char* message)
{
	if(error && check_for_errors)
	{
		fprintf(stderr,"\nAstaroth DSL fatal error: %s\n\n",message);
		exit(EXIT_FAILURE);
	}
}

#include "user_cpu_kernels.h"
#undef  constexpr
#undef size

#include "user_built-in_constants.h"
#include "user_builtin_non_scalar_constants.h"

bool
index_at_boundary(const int x, const int y, const int z)
{
	return  
	      ((x < VAL(AC_nmin).x) || (x >= VAL(AC_nlocal_max).x))
	   || ((y < VAL(AC_nmin).y) || (y >= VAL(AC_nlocal_max).y))
	   || ((z < VAL(AC_nmin).z) || (z >= VAL(AC_nlocal_max).z))
	   ;
}




VertexBufferArray
vbaCreate(const size_t count)
{
  VertexBufferArray vba{};
  memset(&vba, 0, sizeof(vba));

  const size_t bytes = sizeof(vba.on_device.in[0][0]) * count;
  for (size_t i = 0; i < NUM_FIELDS; ++i) {
    vba.on_device.in[i]  = (AcReal*)malloc(bytes);
    vba.on_device.out[i] = (AcReal*)malloc(bytes);
  }
  for (int i = 0; i < NUM_PROFILES; ++i) {
    vba.on_device.profiles.in[i]  = (AcReal*)malloc(bytes);
    vba.on_device.profiles.out[i] = (AcReal*)malloc(bytes);
  }

  vba.on_device.block_factor = (int3){1,1,1};
  return vba;
}

void
vbaDestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_FIELDS; ++i) {
    free(vba->on_device.in[i]);
    free(vba->on_device.out[i]);
    vba->on_device.in[i]  = NULL;
    vba->on_device.out[i] = NULL;
  }
  for (int i = 0; i < NUM_PROFILES; ++i) {
    free(vba->on_device.profiles.in[i]);
    free(vba->on_device.profiles.out[i]);
    vba->on_device.profiles.in[i]  = NULL;
    vba->on_device.profiles.out[i] = NULL;
  }
}
VertexBufferArray VBA = vbaCreate(1000);


void
reset_info_arrays()
{
    memset(stencils_accessed, 0,sizeof(stencils_accessed));
    memset(read_fields,0, sizeof(read_fields));
    memset(field_has_stencil_op,0, sizeof(field_has_stencil_op));
    memset(written_fields, 0,    sizeof(written_fields));
    memset(previous_accessed, 0, sizeof(previous_accessed));
    memset(incoming_ray_value_accessed, 0, sizeof(incoming_ray_value_accessed));
    memset(outgoing_ray_value_accessed, 0, sizeof(outgoing_ray_value_accessed));
    memset(written_complex_fields,0,sizeof(written_complex_fields));
    memset(read_complex_fields,0,sizeof(read_complex_fields));
    std::vector<KernelReduceOutput> empty_vec{};
    reduce_outputs = empty_vec;
    reduce_inputs  = empty_vec;
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
#if AC_DOUBLE_PRECISION
    for(int i = 0; i < NUM_FLOAT_OUTPUTS; ++i)
	    reduced_floats[i] = 0;
#endif
}

void
execute_kernel(const int kernel_index)
{
    reset_info_arrays();
    const Kernel kernel = kernels[kernel_index];
    current_kernel = (AcKernel)kernel_index;
    kernel((int3){0, 0, 0}, (int3){1, 1, 1}, VBA.on_device);
}
int3
get_ghosts()
{
	return (int3)
	{
		d_mesh_info[AC_nmin]
	};
}
void
execute_kernel(const AcKernel kernel_index, const AcBoundary boundary)
{

	const int3 ghosts = get_ghosts();
	reset_info_arrays();
        current_kernel = (AcKernel)kernel_index;
    	const Kernel kernel = kernels[kernel_index];
	if(BOUNDARY_X_BOT & boundary)
	{
		int3 start = (int3){0,ghosts.x,ghosts.y};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}

	if(BOUNDARY_X_TOP & boundary)
	{
		int3 start = (int3){VAL(AC_nlocal).x+ghosts.x, ghosts.y, ghosts.z};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Y_BOT & boundary)
	{
		int3 start = (int3){ghosts.x, 0, ghosts.z};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Y_TOP & boundary)
	{
		int3 start = (int3){ghosts.x, VAL(AC_nlocal).y+ghosts.y, ghosts.z};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Z_BOT  & boundary)
	{
		int3 start = (int3){ghosts.x, ghosts.y, 0};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA.on_device);
	}
	if(BOUNDARY_Z_TOP & boundary)
	{
		int3 start = (int3){ghosts.x, ghosts.y, VAL(AC_nlocal).z+ghosts.z};
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
acAnalysisCheckForDSLErrors(const AcMeshInfo info)
{
	d_mesh_info = info;
	check_for_errors = true;
	for(size_t k = 0; k <NUM_KERNELS; ++k)
	{
    		execute_kernel(k);
	}
	check_for_errors = false;
}


acAnalysisBCInfo 
acAnalysisGetBCInfo(const AcMeshInfo info, const AcKernel bc, const AcBoundary boundary)
{
	bool larger_input  = false;
	bool larger_output = false;
	d_mesh_info = info;
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

KernelAnalysisInfo
acAnalysisGetKernelInfoSingle(const AcMeshInfo info, const AcKernel kernel)
{
	d_mesh_info = info;
	KernelAnalysisInfo res{};
	{
    		execute_kernel(kernel);
    		for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
    		{
    		  for (size_t i = 0; i < NUM_STENCILS; ++i)
    		  {
    		    if (stencils_accessed[j][i])
    		    {
    		      if(i == 0) read_fields[j] |= stencils_accessed[j][i];
    		      field_has_stencil_op[j] |= (i != 0);
    		    }
		    res.stencils_accessed[j][i] |= stencils_accessed[j][i];
    		    read_fields[j] |= previous_accessed[j];
    		  }
    		}
    		for (int j = 0; j < NUM_PROFILES; ++j)
    		  for (size_t i = 0; i < NUM_STENCILS; ++i)
		    res.stencils_accessed[j+NUM_ALL_FIELDS][i] |= stencils_accessed[j+NUM_ALL_FIELDS][i];
		for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
		{
			res.read_fields[i]    = read_fields[i];
			res.field_has_stencil_op[i] = field_has_stencil_op[i];
			res.written_fields[i] = written_fields[i];
			for(int ray = 0; ray < NUM_RAYS; ++ray)
			{
				res.ray_accessed[i][ray] = incoming_ray_value_accessed[i][ray];
				res.ray_accessed[i][ray] |= outgoing_ray_value_accessed[i][ray];
			}
		}
		for(int i = 0; i < NUM_PROFILES; ++i)
		{
			res.read_profiles[i]    = read_profiles[i];
			res.reduced_profiles[i] = reduced_profiles[i];
			res.written_profiles[i] = written_profiles[i];
			//TP: skip value stencil
			for(size_t j = 1; j < NUM_STENCILS; ++j)
				res.profile_has_stencil_op[i] |= stencils_accessed[NUM_ALL_FIELDS+i][j];
		}
		res.n_reduce_outputs = reduce_outputs.size();
		for(size_t i = 0; i < reduce_outputs.size(); ++i)
			res.reduce_outputs[i] = reduce_outputs[i];
		if(res.n_reduce_outputs > NUM_OUTPUTS)
		{
			fprintf(stderr,"Can not reduce variables multiple times in a Kernel\n");
			exit(EXIT_FAILURE);
		}
		
		std::vector<KernelReduceOutput> unique_inputs{};
		for(KernelReduceOutput entry : reduce_inputs)
		{
			if(std::find(unique_inputs.begin(), unique_inputs.end(), entry) == unique_inputs.end())
				unique_inputs.push_back(entry);
		}
		res.n_reduce_inputs = unique_inputs.size();
		for(size_t i = 0; i < unique_inputs.size(); ++i)
			res.reduce_inputs[i] = unique_inputs[i];
	}
	return res;
}
AcResult
acAnalysisGetKernelInfo(const AcMeshInfo info, KernelAnalysisInfo* dst)
{
	d_mesh_info = info;
	memset(dst,0,sizeof(dst[0])*NUM_KERNELS);
	for(size_t k = 0; k <NUM_KERNELS; ++k)
	{
		dst[k] = acAnalysisGetKernelInfoSingle(info,AcKernel(k));
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
  fprintf(fp, "};\n");
}

template <const size_t N, const size_t M>
void
print_info_array(FILE* fp, const char* name, const int arr[NUM_KERNELS][N][M])
{
  fprintf(fp,
          "static int %s[NUM_KERNELS][%ld][%ld] "
          "__attribute__((unused)) =  {",name,N,M);
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    fprintf(fp,"{");
    for (size_t j = 0; j < N; ++j)
    {

        fprintf(fp,"{");
	for(size_t i = 0; i < M; ++ i)
	{
        	fprintf(fp, "%d,", arr[k][j][i]);
	}
    	fprintf(fp,"},");
    }
    fprintf(fp,"},");
  }
  fprintf(fp, "};\n");
}

void
print_info_array(FILE* fp, const char* name, const int arr[NUM_KERNELS][0])
{
  (void)arr;
  fprintf(fp,
          "static int %s[NUM_KERNELS][%d] "
          "__attribute__((unused)) =  {",name,0);
  fprintf(fp, "};\n");
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

  int  write_output[NUM_KERNELS][NUM_ALL_FIELDS]{};
  int  write_complex_output[NUM_KERNELS][NUM_COMPLEX_FIELDS+1]{};
  int  value_complex_output[NUM_KERNELS][NUM_COMPLEX_FIELDS+1]{};
  int  write_profile_output[NUM_KERNELS][NUM_PROFILES]{};
  int  output_previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES]{};
  int  output_incoming_ray_value_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1]{};
  int  output_outgoing_ray_value_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_RAYS+1]{};
  int  output_reduced_profiles[NUM_KERNELS][NUM_PROFILES]{};
  int  output_reduced_reals[NUM_KERNELS][NUM_REAL_OUTPUTS+1]{};
  int  output_reduced_ints[NUM_KERNELS][NUM_INT_OUTPUTS+1]{};
#if AC_DOUBLE_PRECISION
  int  output_reduced_floats[NUM_KERNELS][NUM_FLOAT_OUTPUTS+1]{};
#endif
  int  output_read_profiles[NUM_KERNELS][NUM_PROFILES]{};

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS] "
          "__attribute__((unused)) =  {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
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
      for(int ray = 0; ray < NUM_RAYS; ++ray)
      {
      	output_incoming_ray_value_accessed[k][j][ray] = incoming_ray_value_accessed[j][ray];
      	output_outgoing_ray_value_accessed[k][j][ray] = outgoing_ray_value_accessed[j][ray];
      }
      write_output[k][j] = written_fields[j];
    }
    for(int j = 0; j < NUM_COMPLEX_FIELDS; ++j)
    {
      write_complex_output[k][j] = written_complex_fields[j];
      value_complex_output[k][j] = read_complex_fields[j];
    }
    for(size_t j = 0; j < NUM_PROFILES; ++j)
    {
      output_reduced_profiles[k][j] = reduced_profiles[j];
      output_read_profiles[k][j]    = read_profiles[j];
      write_profile_output[k][j] = written_profiles[j];
    }
    for(int j = 0; j < NUM_REAL_OUTPUTS; ++j)
    	output_reduced_reals[k][j] = reduced_reals[j];
    for(int j = 0; j < NUM_INT_OUTPUTS; ++j)
    	output_reduced_ints[k][j] = reduced_ints[j];
#if AC_DOUBLE_PRECISION
    for(int j = 0; j < NUM_FLOAT_OUTPUTS; ++j)
	output_reduced_floats[k][j] = reduced_floats[j];
#endif

    fprintf(fp,"{");
    for (size_t j = 0; j < NUM_ALL_FIELDS+NUM_PROFILES; ++j)
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
#if AC_DOUBLE_PRECISION
    fwrite(reduced_floats,sizeof(int),NUM_FLOAT_OUTPUTS,fp_floats_reduced);
#endif
  }


  fprintf(fp, "};\n");

  fclose(fp_written_fields);
  fclose(fp_fields_read);
  fclose(fp_field_has_stencil_op);
  fclose(fp_profiles_read);
  fclose(fp_profiles_reduced);

  print_info_array(fp,"previous_accessed",output_previous_accessed);
  print_info_array(fp,"incoming_ray_value_accessed",output_incoming_ray_value_accessed);
  print_info_array(fp,"outgoing_ray_value_accessed",output_outgoing_ray_value_accessed);
  print_info_array(fp,"write_called",write_output);
  print_info_array(fp,"write_complex_called",write_complex_output);
  print_info_array(fp,"value_complex_called",value_complex_output);
  print_info_array(fp,"write_called_profile",write_profile_output);
  print_info_array(fp,"reduced_profiles",output_reduced_profiles);
  print_info_array(fp,"read_profiles",output_read_profiles);
  print_info_array(fp,"reduced_reals",output_reduced_reals);
  print_info_array(fp,"reduced_ints",output_reduced_ints);
#if AC_DOUBLE_PRECISION
  print_info_array(fp,"reduced_floats",output_reduced_floats);
#endif
  fprintf(fp,"const bool has_mem_access_info __attribute__((unused)) = true;\n");

  fclose(fp);


#include "gmem_arrays_output_accesses.h"
  fprintf(stderr,"Generated stencil accesses\n");
  return EXIT_SUCCESS;
}
#endif
