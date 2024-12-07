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


#define DEVICE_INLINE
#ifndef AC_IN_AC_LIBRARY
#define AC_IN_AC_LIBRARY
#endif
#define longlong long long
#include "func_attributes.h"

#include <assert.h>
/*
#if AC_USE_HIP || __HIP_PLATFORM_HCC__ // Hack to ensure hip is used even if
                                       // USE_HIP is not propagated properly
                                       // TODO figure out better way
#include <hip/hip_runtime.h>           // Needed in files that include kernels
#else
#include <cuda_runtime_api.h>
#endif
*/
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
#define blockIdx ((int3){0, 0, 0})
#define blockDim ((int3){0, 0, 0})
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
	return 0;
}



#define idx  ((int)IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z))

#endif
#undef  __device__
#define __device__

#undef  __constant__
#define __constant__

#include "math_utils.h"
 
#define constexpr
#include "acc_runtime.h"
#undef constexpr

static int stencils_accessed[NUM_ALL_FIELDS][NUM_STENCILS]{{}};
static int previous_accessed[NUM_ALL_FIELDS]{};
static int written_fields[NUM_ALL_FIELDS]{};
static int read_fields[NUM_ALL_FIELDS]{};
static int field_has_stencil_op[NUM_ALL_FIELDS]{};
static int read_profiles[NUM_PROFILES]{};
static int reduced_profiles[NUM_PROFILES]{};
static int written_profiles[NUM_PROFILES]{};

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
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_SUM,current_kernel});
}

void
reduce_max_real(const bool&, const AcReal, const AcRealOutputParam dst)
{
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MAX,current_kernel});
}

void
reduce_min_real(const bool&, const AcReal, const AcRealOutputParam dst)
{
	reduce_outputs.push_back({(int)dst,AC_REAL_TYPE,REDUCE_MIN,current_kernel});
}

void
reduce_sum_int(const bool&, const AcReal, const AcIntOutputParam dst)
{
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_SUM,current_kernel});
}

void
reduce_max_int(const bool&, const AcReal, const AcIntOutputParam dst)
{
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MAX,current_kernel});
}

void
reduce_min_int(const bool&, const AcReal, const AcIntOutputParam dst)
{
	reduce_outputs.push_back({(int)dst,AC_INT_TYPE,REDUCE_MIN,current_kernel});
}

void
reduce_prof(const bool&, const AcReal, const Profile dst)
{
	if constexpr (NUM_PROFILES != 0)
		reduced_profiles[(int)dst] |= 1;
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
typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);
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
#include "user_cpu_kernels.h"
#undef  constexpr
#undef size



VertexBufferArray
vbaCreate(const size_t count)
{
  VertexBufferArray vba{};
  memset(&vba, 0, sizeof(vba));

  const size_t bytes = sizeof(vba.in[0][0]) * count;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    vba.in[i]  = (AcReal*)malloc(bytes);
    vba.out[i] = (AcReal*)malloc(bytes);
  }
  for (size_t i = 0; i < NUM_PROFILES; ++i) {
    vba.profiles.in[i]  = (AcReal*)malloc(bytes);
    vba.profiles.out[i] = (AcReal*)malloc(bytes);
  }

  return vba;
}

void
vbaDestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    free(vba->in[i]);
    free(vba->out[i]);
    vba->in[i]  = NULL;
    vba->out[i] = NULL;
  }
  for (size_t i = 0; i < NUM_PROFILES; ++i) {
    free(vba->profiles.in[i]);
    free(vba->profiles.out[i]);
    vba->profiles.in[i]  = NULL;
    vba->profiles.out[i] = NULL;
  }
}
VertexBufferArray VBA = vbaCreate(1000);

void
execute_kernel(const int kernel_index)
{
    const Kernel kernel = kernels[kernel_index];
    kernel((int3){0, 0, 0}, (int3){1, 1, 1}, VBA);
}
void
execute_kernel(const AcKernel kernel_index, const AcBoundary boundary)
{
    	const Kernel kernel = kernels[kernel_index];
	if(BOUNDARY_X_BOT & boundary)
	{
		int3 start = (int3){0,NGHOST,NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}

	if(BOUNDARY_X_TOP & boundary)
	{
		int3 start = (int3){VAL(AC_nlocal).x+NGHOST, NGHOST, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
	if(BOUNDARY_Y_BOT & boundary)
	{
		int3 start = (int3){NGHOST, 0, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
	if(BOUNDARY_Y_TOP & boundary)
	{
		int3 start = (int3){NGHOST, VAL(AC_nlocal).y+NGHOST, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
	if(BOUNDARY_Z_BOT  & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, 0};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
	if(BOUNDARY_Z_TOP & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, VAL(AC_nlocal).z+NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
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
		current_kernel = (AcKernel)k;
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

#ifndef AC_NO_MAIN
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

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS] "
          "__attribute__((unused)) =  {");
  int  write_output[NUM_KERNELS][NUM_ALL_FIELDS]{};
  int  output_previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS]{};
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
  }


  fprintf(fp, "};");

  fclose(fp_written_fields);
  fclose(fp_fields_read);
  fclose(fp_field_has_stencil_op);
  fclose(fp_profiles_read);
  fclose(fp_profiles_reduced);

  fprintf(fp,
          "static int previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] "
          "__attribute__((unused)) =  {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    fprintf(fp,"{");
    for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
        fprintf(fp, "%d,", output_previous_accessed[k][j]);
    fprintf(fp,"},");
  }
  fprintf(fp, "};");

  fprintf(fp,
          "static int write_called[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] "
          "__attribute__((unused)) =  {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    fprintf(fp,"{");
    for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
        fprintf(fp, "%d,", write_output[k][j]);
    fprintf(fp,"},");
  }
  fprintf(fp, "};");
  fclose(fp);

#include "gmem_arrays_output_accesses.h"
  return EXIT_SUCCESS;
}
#endif
