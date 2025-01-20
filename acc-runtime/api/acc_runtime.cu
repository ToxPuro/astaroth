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

#include <vector> // tbconfig

#include "errchk.h"
#include "math_utils.h"

#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels
#endif

#define USE_COMPRESSIBLE_MEMORY (0)

#include "acc/implementation.h"

static dim3 last_tpb = (dim3){0, 0, 0};

Volume
acKernelLaunchGetLastTPB(void)
{
  return to_volume(last_tpb);
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

// Astaroth 2.0 backwards compatibility START
#define d_multigpu_offset (d_mesh_info.int3_params[AC_multigpu_offset])

int __device__ __forceinline__
DCONST(const AcIntParam param)
{
  return d_mesh_info.int_params[param];
}
int3 __device__ __forceinline__
DCONST(const AcInt3Param param)
{
  return d_mesh_info.int3_params[param];
}
AcReal __device__ __forceinline__
DCONST(const AcRealParam param)
{
  return d_mesh_info.real_params[param];
}
AcReal3 __device__ __forceinline__
DCONST(const AcReal3Param param)
{
  return d_mesh_info.real3_params[param];
}

#define DEVICE_VTXBUF_IDX(i, j, k)                                             \
  ((i) + (j) * DCONST(AC_mx) + (k) * DCONST(AC_mxy))

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

#define Field3(x, y, z) make_int3((x), (y), (z))
#define print printf                          // TODO is this a good idea?
#define len(arr) sizeof(arr) / sizeof(arr[0]) // Leads to bugs if the user
// passes an array into a device function and then calls len (need to modify
// the compiler to always pass arrays to functions as references before
// re-enabling)

#include "random.cuh"

#include "user_kernels.h"

typedef struct {
  Kernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;

static TBConfig getOptimalTBConfig(const Kernel kernel, const int3 dims,
                                   VertexBufferArray vba);

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
acPBAReset(const cudaStream_t stream, ProfileBufferArray* pba)
{
  // Set pba.in data to all-nan and pba.out to 0
  for (int i = 0; i < NUM_PROFILES; ++i) {
    acKernelFlush(stream, pba->in[i], pba->count, (AcReal)NAN);
    acKernelFlush(stream, pba->out[i], pba->count, (AcReal)0.0);
  }
  return AC_SUCCESS;
}

ProfileBufferArray
acPBACreate(const size_t count)
{
  ProfileBufferArray pba = {.count = count};

  const size_t bytes = sizeof(pba.in[0][0]) * pba.count * NUM_PROFILES;
  AcReal *in, *out;
#if USE_COMPRESSIBLE_MEMORY
  ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)&in, bytes));
  ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)&out, bytes));
#else
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&in, bytes));
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&out, bytes));
#endif
  for (int i = 0; i < NUM_PROFILES; ++i) {
    pba.in[i]  = &in[i * pba.count];
    pba.out[i] = &out[i * pba.count];
  }

  acPBAReset(0, &pba);
  cudaDeviceSynchronize();
  return pba;
}

void
acPBADestroy(ProfileBufferArray* pba)
{
#if USE_COMPRESSIBLE_MEMORY
  freeCompressible(pba->in[0],
                   sizeof(pba.in[0][0]) * pba->count * NUM_PROFILES);
  freeCompressible(pba->out[0],
                   sizeof(pba.out[0][0]) * pba->count * NUM_PROFILES);
#else
  cudaFree(pba->in[0]);
  cudaFree(pba->out[0]);
#endif
  for (int i = 0; i < NUM_PROFILES; ++i) {
    pba->in[i]  = NULL;
    pba->out[i] = NULL;
  }
  pba->count = 0;
}

AcResult
acVBAReset(const cudaStream_t stream, VertexBufferArray* vba)
{
  const size_t count = vba->bytes / sizeof(vba->in[0][0]);

  // Set vba.in data to all-nan and vba.out to 0
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    acKernelFlush(stream, vba->in[i], count, (AcReal)NAN);
    acKernelFlush(stream, vba->out[i], count, (AcReal)0.0);
  }

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBAReset(stream, &vba->profiles);
  return AC_SUCCESS;
}

VertexBufferArray
acVBACreate(const size_t mx, const size_t my, const size_t mz)
{
  VertexBufferArray vba;

  const size_t bytes = sizeof(vba.in[0][0]) * mx * my * mz;
  vba.bytes          = bytes;
  vba.mx             = mx;
  vba.my             = my;
  vba.mz             = mz;

  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
#if USE_COMPRESSIBLE_MEMORY
    ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)&vba.in[i], bytes));
    ERRCHK_CUDA_ALWAYS(mallocCompressible((void**)&vba.out[i], bytes));
#else
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&vba.in[i], bytes));
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&vba.out[i], bytes));
#endif
  }

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  vba.profiles = acPBACreate(mz);

  acVBAReset(0, &vba);
  cudaDeviceSynchronize();
  return vba;
}

void
acVBADestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
#if USE_COMPRESSIBLE_MEMORY
    freeCompressible(vba->in[i], vba->bytes);
    freeCompressible(vba->out[i], vba->bytes);
#else
    cudaFree(vba->in[i]);
    cudaFree(vba->out[i]);
#endif
    vba->in[i]  = NULL;
    vba->out[i] = NULL;
  }
  vba->bytes = 0;
  vba->mx    = 0;
  vba->my    = 0;
  vba->mz    = 0;

  // Note: should be moved out when refactoring VBA to KernelParameterArray
  acPBADestroy(&vba->profiles);
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
         static_cast<double>(milliseconds));

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

#define GEN_LOAD_UNIFORM(LABEL_UPPER, LABEL_LOWER)                             \
  ERRCHK_ALWAYS(param < NUM_##LABEL_UPPER##_PARAMS);                           \
  cudaDeviceSynchronize(); /* See note in acLoadStencil */                     \
                                                                               \
  const size_t offset = (size_t) &                                             \
                        d_mesh_info.LABEL_LOWER##_params[param] - (size_t) &   \
                        d_mesh_info;                                           \
                                                                               \
  const cudaError_t retval = cudaMemcpyToSymbol(                               \
      d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice);     \
  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;

AcResult
acLoadRealUniform(const cudaStream_t /* stream */, const AcRealParam param,
                  const AcReal value)
{
  if (isnan(value)) {
    fprintf(stderr,
            "WARNING: Passed an invalid value %g to device constant %s. "
            "Skipping.\n",
            (double)value, realparam_names[param]);
    return AC_FAILURE;
  }
  GEN_LOAD_UNIFORM(REAL, real);
}

AcResult
acLoadReal3Uniform(const cudaStream_t /* stream */, const AcReal3Param param,
                   const AcReal3 value)
{
  if (isnan(value.x) || isnan(value.y) || isnan(value.z)) {
    fprintf(stderr,
            "WARNING: Passed an invalid value (%g, %g, %g) to device constant "
            "%s. Skipping.\n",
            (double)value.x, (double)value.y, (double)value.z,
            real3param_names[param]);
    return AC_FAILURE;
  }
  GEN_LOAD_UNIFORM(REAL3, real3);
}

AcResult
acLoadIntUniform(const cudaStream_t /* stream */, const AcIntParam param,
                 const int value)
{
  GEN_LOAD_UNIFORM(INT, int);
}

AcResult
acLoadInt3Uniform(const cudaStream_t /* stream */, const AcInt3Param param,
                  const int3 value)
{
  GEN_LOAD_UNIFORM(INT3, int3);
}

#define GEN_STORE_UNIFORM(LABEL_UPPER, LABEL_LOWER)                            \
  ERRCHK_ALWAYS(param < NUM_##LABEL_UPPER##_PARAMS);                           \
  cudaDeviceSynchronize(); /* See notes in GEN_LOAD_UNIFORM */                 \
                                                                               \
  const size_t offset = (size_t) &                                             \
                        d_mesh_info.LABEL_LOWER##_params[param] - (size_t) &   \
                        d_mesh_info;                                           \
                                                                               \
  const cudaError_t retval = cudaMemcpyFromSymbol(                             \
      value, d_mesh_info, sizeof(*value), offset, cudaMemcpyDeviceToHost);     \
  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;

AcResult
acStoreRealUniform(const cudaStream_t /* stream */, const AcRealParam param,
                   AcReal* value)
{
  GEN_STORE_UNIFORM(REAL, real);
}

AcResult
acStoreReal3Uniform(const cudaStream_t /* stream */, const AcReal3Param param,
                    AcReal3* value)
{
  GEN_STORE_UNIFORM(REAL3, real3);
}

AcResult
acStoreIntUniform(const cudaStream_t /* stream */, const AcIntParam param,
                  int* value)
{
  GEN_STORE_UNIFORM(INT, int);
}

AcResult
acStoreInt3Uniform(const cudaStream_t /* stream */, const AcInt3Param param,
                   int3* value)
{
  GEN_STORE_UNIFORM(INT3, int3);
}

static TBConfig
autotune(const Kernel kernel, const int3 dims, VertexBufferArray vba)
{
  size_t id = (size_t)-1;
  for (size_t i = 0; i < NUM_KERNELS; ++i) {
    if (kernels[i] == kernel) {
      id = i;
      break;
    }
  }
  ERRCHK_ALWAYS(id < NUM_KERNELS);
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

  const int3 start = (int3){
      STENCIL_ORDER / 2,
      STENCIL_ORDER / 2,
      STENCIL_ORDER / 2,
  };
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

  FILE* fp = fopen("autotune.csv", "a");
  ERRCHK_ALWAYS(fp);
#if IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
  fprintf(fp, "%d, (%d, %d, %d), (%d, %d, %d), %g\n", IMPLEMENTATION, nx, ny,
          nz, best_tpb.x, best_tpb.y, best_tpb.z,
          (double)best_time / num_iters);
#else
  fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %g\n", IMPLEMENTATION, dims.x,
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

static TBConfig
getOptimalTBConfig(const Kernel kernel, const int3 dims, VertexBufferArray vba)
{
  for (auto c : tbconfigs) {
    if (c.kernel == kernel && c.dims == dims)
      return c;
  }
  TBConfig c = autotune(kernel, dims, vba);
  tbconfigs.push_back(c);
  return c;
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
acLoadMeshInfo(const AcMeshInfo info, const cudaStream_t stream)
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

static __host__ __device__ constexpr AcIndex
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

static __host__ __device__ constexpr AcIndex
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

  reindex<<<bpg, tpb, 0, stream>>>(in, in_offset, in_shape, //
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

static __global__ void
reindex_cross(const CrossProductArrays arrays, const AcIndex in_offset,
              const AcShape in_shape, const AcIndex out_offset,
              const AcShape out_shape, const AcShape block_shape)
{
  const AcIndex idx = to_spatial(static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x, block_shape);

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
      const AcReal3 res                           = cross(a, b);
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

AcResult
acReindexCross(const cudaStream_t stream, //
               const VertexBufferArray vba, const AcIndex in_offset,
               const AcShape in_shape, //
               AcReal* out, const AcIndex out_offset, const AcShape out_shape,
               const AcShape block_shape)
{
#if 0 // ifdef AC_TFM_ENABLED
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

  reindex_cross<<<bpg, tpb, 0, stream>>>(arrays, in_offset, in_shape,
                                         out_offset, out_shape, block_shape);
  return AC_SUCCESS;
#else
  ERROR("acReindexCross called but AC_TFM_ENABLED was false");
  (void)stream; // Unused
  (void)vba; // Unused
  (void)in_offset; // Unused
  (void)in_shape; // Unused
  (void)out; // Unused
  (void)out_offset; // Unused
  (void)out_shape; // Unused
  (void)block_shape; // Unused
  (void)reindex_cross; // Unused
  return AC_FAILURE;
#endif
}

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
    // printf("Offset %zu: %zu\n", i, offsets[i]);
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
  multiply_inplace<<<bpg, tpb>>>(value, count, array);
  ERRCHK_CUDA_KERNEL();
  ERRCHK_CUDA(cudaDeviceSynchronize()); // NOTE: explicit sync here for safety
  return AC_SUCCESS;
}

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
