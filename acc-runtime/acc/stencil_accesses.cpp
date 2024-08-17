#define AC_RUNTIME_SOURCE

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
#include "user_defines.h"

typedef struct
{
	VertexBufferHandle x;
	VertexBufferHandle y;
	VertexBufferHandle z;
} Field3;

#include <array>

#undef __device__
#define __device__
#undef __global__
#define __global__
#undef __launch_bounds__
#define __launch_bounds__(x, y)
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
#define print printf
#define len(arr) sizeof(arr) / sizeof(arr[0])
#define rand_uniform()        (0.5065983774206012); // Chosen by a fair dice roll
#define random_uniform(TID) (0.5065983774206012); // Chosen by a fair dice roll
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
reduce_sum(const bool& condition, const AcReal& val, const AcRealOutputParam& output){}
void 
reduce_min(const bool& condition, const AcReal& val, const AcRealOutputParam& output){}
void 
reduce_max(const bool& condition, const AcReal& val, const AcRealOutputParam& output){}

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
constexpr Field3 
MakeField3(const Field& x, const Field& y, const Field& z)
{
	return (Field3){x,y,z};
}
template <size_t N>
constexpr __device__ __forceinline__
std::array<Field3,N>
MakeField3(const Field (&x)[N], const Field (&y)[N], const Field (&z)[N])
{
	std::array<int3,N> res{};
	for(size_t i = 0; i < N; ++i)
		res[i] = (Field3){x,y,z};
	return res;
}
#include "dconst_accesses_decl.h"


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
__ballot(bool val)
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
#include "user_constants.h"
#include "dconst_arrays_decl.h"
#define DECLARE_GMEM_ARRAY(DATATYPE, DEFINE_NAME, ARR_NAME) DATATYPE gmem_##DEFINE_NAME##_arrays[NUM_##ARR_NAME##_ARRAYS+1][1000] {}
#include "gmem_arrays_decl.h"

AcReal smem[8 * 1024 * 1024]; // NOTE: arbitrary limit: need to allocate at
                              // least the max smem size of the device
static AcReal3 AC_INTERNAL_global_real_vec = {0.0,0.0,0.0};
static int3 AC_INTERNAL_global_int_vec = {0,0,0};

static int stencils_accessed[NUM_FIELDS][NUM_STENCILS] = {{0}};
static int previous_accessed[NUM_FIELDS] = {0};
static int written_fields[NUM_FIELDS] = {0};
#include "analysis_stencils.h"
void
write_base (const Field& field, const AcReal& value)
{
	written_fields[field]=1;
}
void
write_to_index (const Field& field, const int& index_out, const AcReal& value)
{
	written_fields[field]=1;
}
AcReal
previous_base(const Field& field)
{
	previous_accessed[field] = 1;
	return AcReal(1.0);
}
AcReal
AC_INTERNAL_read_field(const Field& field, const AcReal& val)
{
	(void)val;
	stencils_accessed[field][stencil_value_stencil] = 1;
	return AcReal(1.0);
}
static AcMeshInfo d_mesh_info{};
#define suppress_unused_warning(X) (void)X

static std::vector<int> executed_conditionals{};
#define constexpr
#include "user_kernels.h"
#undef  constexpr

#include "extern_kernels.h"



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
}
void
execute_kernel(const int kernel)
{
    VertexBufferArray vba = vbaCreate(1000);
    kernels[kernel]((int3){NGHOST, NGHOST, NGHOST}, (int3){NGHOST+1, NGHOST+1, NGHOST+1}, vba);
    vbaDestroy(&vba);
}
int
get_executed_conditionals()
{
  executed_conditionals = {};
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
	  execute_kernel(k);
  }
  FILE* fp_executed_conditionals = fopen("executed_conditionals.bin", "wb");
  const int size = executed_conditionals.size();
  fwrite(&size, sizeof(int), 1, fp_executed_conditionals);
  fwrite(executed_conditionals.data(), sizeof(int), executed_conditionals.size(), fp_executed_conditionals);
  fclose(fp_executed_conditionals);
  return EXIT_SUCCESS;
}	

int
main(int argc, char* argv[])
{
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
  static int read_fields[NUM_FIELDS];
  static int field_has_stencil_op[NUM_FIELDS];


  fprintf(fp,
          "static int stencils_accessed[NUM_KERNELS][NUM_FIELDS][NUM_STENCILS] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    memset(stencils_accessed, 0,
           sizeof(stencils_accessed[0][0]) * NUM_FIELDS * NUM_STENCILS);
    memset(read_fields,0, sizeof(read_fields[0]) * NUM_FIELDS);
    memset(field_has_stencil_op,0, sizeof(field_has_stencil_op[0]) * NUM_FIELDS);
    memset(written_fields, 0,
           sizeof(written_fields[0]) * NUM_FIELDS);
    execute_kernel(k);
    for (size_t j = 0; j < NUM_FIELDS; ++j)
    {
      for (size_t i = 0; i < NUM_STENCILS; ++i)
      {
        if (stencils_accessed[j][i])
	{
	  read_fields[j] = (i == 0);
	  field_has_stencil_op[j] = (i != 0);
          fprintf(fp, "[%lu][%lu][%lu] = 1,", k, j, i);
	}
      }
    }
    fwrite(read_fields,sizeof(int), NUM_FIELDS,fp_fields_read);
    fwrite(field_has_stencil_op,sizeof(int), NUM_FIELDS,fp_field_has_stencil_op);
    fwrite(written_fields,sizeof(int),NUM_FIELDS,fp_written_fields);
  }


  fprintf(fp, "};");
  fclose(fp_written_fields);
  fclose(fp_fields_read);
  fclose(fp_field_has_stencil_op);

  fprintf(fp,
          "static int previous_accessed[NUM_KERNELS][NUM_FIELDS] "
          "= {");
  for (size_t k = 0; k < NUM_KERNELS; ++k) {
    memset(previous_accessed, 0,
           sizeof(previous_accessed[0]) * NUM_FIELDS);
    VertexBufferArray vba = vbaCreate(1000);
    kernels[k]((int3){0, 0, 0}, (int3){1, 1, 1}, vba);
    vbaDestroy(&vba);

    for (size_t j = 0; j < NUM_FIELDS; ++j)
        if (previous_accessed[j])
          fprintf(fp, "[%lu][%lu] = 1,", k, j);
  }
  fprintf(fp, "};");


  fclose(fp);


  return EXIT_SUCCESS;
}
