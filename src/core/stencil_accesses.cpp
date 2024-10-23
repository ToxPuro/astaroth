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

#if TWO_D == 0

#define globalGridN ((int3) {NGHOST*2, NGHOST*2,NGHOST*2})
#else

#define globalGridN ((int3) {NGHOST*2, NGHOST*2,1})
#endif


#if TWO_D == 0
#define localCompdomainVertexIdx ((int3){vertexIdx.x - (STENCIL_WIDTH-1)/2, vertexIdx.y - (STENCIL_HEIGHT-1)/2, vertexIdx.y - (STENCIL_DEPTH-1)/2})
#else
#define localCompdomainVertexIdx ((int3){vertexIdx.x - (STENCIL_WIDTH-1)/2, vertexIdx.y - (STENCIL_HEIGHT-1)/2, 0})
#endif

#define local_compdomain_idx ((LOCAL_COMPDOMAIN_IDX(localCompdomainVertexIdx))

void 
reduce_sum_real(const bool&, const AcReal&, const AcRealOutputParam&){}
void 
reduce_min_real(const bool&, const AcReal&, const AcRealOutputParam&){}
void 
reduce_max_real(const bool&, const AcReal&, const AcRealOutputParam&){}
void 
reduce_sum_real_x(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_y(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_z(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_xy(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_xz(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_yx(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_yz(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_zx(const bool&, const AcReal&, const Profile&){}
void 
reduce_sum_real_zy(const bool&, const AcReal&, const Profile&){}


void 
reduce_sum_int(const bool&,  const int&, const AcIntOutputParam&){}
void 
reduce_min_int(const bool&,  const int&, const AcIntOutputParam&){}
void 
reduce_max_int(const bool&,  const int&, const AcIntOutputParam&){}

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

int
IDX(const int3 idx)
{
  (void)idx; // Unused
  return 0;
}

int
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



#define idx ((int)IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z))

#endif
#undef  __device__
#define __device__

#undef  __constant__
#define __constant__

#include "math_utils.h"
 
#include "acc_runtime.h"
extern "C" 
{
	AcResult acAnalysisGetKernelInfo(const AcMeshInfo info, KernelAnalysisInfo* src);
	acAnalysisBCInfo acAnalysisGetBCInfo(const AcMeshInfo info, const AcKernel bc, const AcBoundary boundary);
}
//#include "user_constants.h"
typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);
#include "user_kernel_declarations.h"

constexpr AcMeshInfo
get_d_mesh_info()
{
	//TP: DCONST ints and bools have to be evaluated to 1 since PC depends on conditionals like if(int) and if(bool) being true at analysis 
	AcMeshInfo res{};
  	for(int i = 0; i < NUM_INT_PARAMS; ++i)
	  res.int_params[i] = 1;
  	for(int i = 0; i < NUM_BOOL_PARAMS; ++i)
	  res.bool_params[i] = true;
	return res;
}

static AcMeshInfo d_mesh_info = get_d_mesh_info();

AcResult
acAnalysisLoadMeshInfo(const AcMeshInfo info) 
{d_mesh_info = info; return AC_SUCCESS;}

#include "dconst_decl.h"
#include "rconst_decl.h"

#include "dconst_arrays_decl.h"
#include "gmem_arrays_accessed_decl.h"
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME) static const DATATYPE ARR_NAME##return_var{}; \
							    struct tmp_struct_##ARR_NAME {const DATATYPE& operator[](const int) {gmem_##DEFINE_NAME##_arrays_accessed[ARR_NAME] = 1; return ARR_NAME##return_var;}}; \
							    static tmp_struct_##ARR_NAME AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME {};

#define DECLARE_CONST_DIMS_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME, DIMS) static const DATATYPE ARR_NAME##return_var{}; \
							    struct tmp_struct_##ARR_NAME {const DATATYPE& operator[](const int) {gmem_##DEFINE_NAME##_arrays_accessed[ARR_NAME] = 1; return ARR_NAME##return_var;}}; \
							    static tmp_struct_##ARR_NAME AC_INTERNAL_gmem_##DEFINE_NAME##_arrays_##ARR_NAME {};
#include "gmem_arrays_decl.h"

AcReal smem[8 * 1024 * 1024]; // NOTE: arbitrary limit: need to allocate at
                              // least the max smem size of the device
[[maybe_unused]] static AcReal3 AC_INTERNAL_global_real_vec = {0.0,0.0,0.0};
[[maybe_unused]] static int3 AC_INTERNAL_global_int_vec = {0,0,0};

static int stencils_accessed[NUM_ALL_FIELDS][NUM_STENCILS]{{}};
static int previous_accessed[NUM_ALL_FIELDS]{};
static int written_fields[NUM_ALL_FIELDS]{};
static int read_fields[NUM_ALL_FIELDS]{};
static int field_has_stencil_op[NUM_ALL_FIELDS]{};
#include "analysis_stencils.h"
[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_WRITE    = (1 << 0);
[[maybe_unused]] constexpr int AC_IN_BOUNDS_WRITE = (1 << 1);
[[maybe_unused]] constexpr int AC_ALL_WRITE      = ~0;

[[maybe_unused]] constexpr int AC_OUT_OF_BOUNDS_READ     = (1 << 0);
[[maybe_unused]] constexpr int AC_IN_BOUNDS_READ  = (1 << 1);
[[maybe_unused]] constexpr int AC_ALL_READ      = ~0;


AcReal
VAL(const AcRealParam& param)
{
	return DCONST(param);
}

AcReal
VAL(const AcReal& val)
{
	return val;
}

int
VAL(const AcIntParam& param)
{
	return DCONST(param);
}

int
VAL(const int& val)
{
	return val;
}

bool
index_at_boundary(const int x, const int y, const int z)
{
#if TWO_D
	(void)z;
#endif
	return  
	      ((x < NGHOST) || (x >= VAL(AC_nx_max)))
	   || ((y < NGHOST) || (y >= VAL(AC_ny_max)))
#if TWO_D == 0
	   || ((z < NGHOST) || (z >= VAL(AC_nz_max)))
#endif
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
	written_fields[field] |= AC_OUT_OF_BOUNDS_WRITE;
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

static std::vector<int> executed_conditionals{};
#define constexpr
#include "user_cpu_kernels.h"
#undef  constexpr



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
		int3 start = (int3){AC_nx+NGHOST, NGHOST, NGHOST};
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
		int3 start = (int3){NGHOST, AC_ny+NGHOST, NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
#if TWO_D == 0
	if(BOUNDARY_Z_BOT  & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, 0};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
	if(BOUNDARY_Z_TOP & boundary)
	{
		int3 start = (int3){NGHOST, NGHOST, AC_nz+NGHOST};
		int3 end   = start + (int3){1,1,1};
    		kernel(start,end,VBA);
	}
#endif
}
int
get_executed_conditionals()
{ 
  fprintf(stderr,"Getting executed conditionals\n");
  fflush(stderr);
  executed_conditionals = {};
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
	  execute_kernel(k);
  }
  FILE* fp_executed_conditionals = fopen("executed_conditionals.bin", "wb");
  const int size = executed_conditionals.size();
  fwrite(&size, sizeof(int), 1, fp_executed_conditionals);
  fwrite(executed_conditionals.data(), sizeof(int), executed_conditionals.size(), fp_executed_conditionals);
  fclose(fp_executed_conditionals);
  fprintf(stderr,"Got executed conditionals\n");
  fflush(stderr);
  return EXIT_SUCCESS;
}	

void
reset_info_arrays()
{
    memset(stencils_accessed, 0,
           sizeof(stencils_accessed[0][0]) * NUM_ALL_FIELDS * NUM_STENCILS);
    memset(read_fields,0, sizeof(read_fields[0]) * NUM_ALL_FIELDS);
    memset(field_has_stencil_op,0, sizeof(field_has_stencil_op[0]) * NUM_ALL_FIELDS);
    memset(written_fields, 0,    sizeof(written_fields[0]) * NUM_ALL_FIELDS);
    memset(previous_accessed, 0, sizeof(previous_accessed[0]) * NUM_ALL_FIELDS);
}

acAnalysisBCInfo 
acAnalysisGetBCInfo(const AcMeshInfo info, const AcKernel bc, const AcBoundary boundary)
{
	bool larger_input  = false;
	bool larger_output = false;
	d_mesh_info = info;
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
acAnalysisGetKernelInfo(const AcMeshInfo info, KernelAnalysisInfo* src)
{
	d_mesh_info = info;
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
    			    read_fields[j] |= previous_accessed[j];
    			  }
    			}
    		}
		for(size_t i = 0; i < NUM_ALL_FIELDS; ++i)
		{
			src->read_fields[k][i]    = read_fields[i];
			src->field_has_stencil_op[k][i] = field_has_stencil_op[i];
			src->written_fields[k][i] = written_fields[i];
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
  if(!strcmp(argv[1],"-C")) return get_executed_conditionals();
  const char* output = argv[1];
  FILE* fp           = fopen(output, "w+");
  assert(fp);

  FILE* fp_fields_read = fopen("user_read_fields.bin","wb");
  FILE* fp_written_fields = fopen("user_written_fields.bin", "wb");
  FILE* fp_field_has_stencil_op = fopen("user_field_has_stencil_op.bin","wb");

  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES][NUM_STENCILS] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
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
    	      fprintf(fp, "[%lu][%lu][%lu] = 1,", k, j, i);
    	    }
    	    read_fields[j] |= previous_accessed[j];
    	  }
    	}
    }


    
    fwrite(read_fields,sizeof(int), NUM_ALL_FIELDS,fp_fields_read);
    fwrite(field_has_stencil_op,sizeof(int), NUM_ALL_FIELDS,fp_field_has_stencil_op);
    fwrite(written_fields,sizeof(int),NUM_ALL_FIELDS,fp_written_fields);
  }


  fprintf(fp, "};");

  fclose(fp_written_fields);
  fclose(fp_fields_read);
  fclose(fp_field_has_stencil_op);

  fprintf(fp,
          "static int previous_accessed[NUM_KERNELS][NUM_ALL_FIELDS+NUM_PROFILES] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    memset(previous_accessed, 0,
           sizeof(previous_accessed[0]) * NUM_ALL_FIELDS);
    VertexBufferArray vba = vbaCreate(1000);
    kernels[k]((int3){0, 0, 0}, (int3){1, 1, 1}, vba);
    vbaDestroy(&vba);

    for (size_t j = 0; j < NUM_ALL_FIELDS; ++j)
        if (previous_accessed[j])
          fprintf(fp, "[%lu][%lu] = 1,", k, j);
  }
  fprintf(fp, "};");


  fclose(fp);

#include "gmem_arrays_output_accesses.h"


  return EXIT_SUCCESS;
}
#endif
