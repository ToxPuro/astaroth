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
#define USE_SMEM (0) // Remember to modify USE_SMEM also in stencilgen.cc
#if USE_SMEM
static const size_t pad = 0;
#endif

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
  ((i) + (j)*DCONST(AC_mx) + (k)*DCONST(AC_mxy))

__device__ constexpr int
IDX(const int i)
{
  return i;
}

__device__ __forceinline__ int
IDX(const int i, const int j, const int k)
{
  return DEVICE_VTXBUF_IDX(i, j, k);
}

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
flush_kernel(AcReal* arr, const size_t n)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n)
    arr[idx] = (AcReal)NAN;
}

AcResult
acKernelFlush(AcReal* arr, const size_t n)
{
  const size_t tpb = 256;
  const size_t bpg = (size_t)(ceil((double)n / tpb));
  flush_kernel<<<bpg, tpb>>>(arr, n);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  return AC_SUCCESS;
}

VertexBufferArray
acVBACreate(const size_t count)
{
  VertexBufferArray vba;

  const size_t bytes = sizeof(vba.in[0][0]) * count;
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&vba.in[i], bytes));
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&vba.out[i], bytes));

    // Set vba.in data to all-nan and vba.out to 0
    acKernelFlush(vba.in[i], count);
    ERRCHK_CUDA_ALWAYS(cudaMemset((void*)vba.out[i], 0, bytes));
  }

  return vba;
}

void
acVBADestroy(VertexBufferArray* vba)
{
  for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
    cudaFree(vba->in[i]);
    cudaFree(vba->out[i]);
    vba->in[i]  = NULL;
    vba->out[i] = NULL;
  }
}

AcResult
acLaunchKernel(Kernel kernel, const cudaStream_t stream, const int3 start,
               const int3 end, VertexBufferArray vba)
{
  const int3 n = end - start;

  const dim3 tpb = getOptimalTBConfig(kernel, n, vba).tpb;
  const dim3 bpg((unsigned int)ceil(n.x / double(tpb.x)), //
                 (unsigned int)ceil(n.y / double(tpb.y)), //
                 (unsigned int)ceil(n.z / double(tpb.z)));
#if USE_SMEM
  const size_t smem = (tpb.x + STENCIL_WIDTH - 1 + pad) *
                      (tpb.y + STENCIL_HEIGHT - 1) * sizeof(AcReal);
#else
  const size_t smem = 0;
#endif

  kernel<<<bpg, tpb, smem, stream>>>(start, end, vba);
  ERRCHK_CUDA_KERNEL();

  return AC_SUCCESS;
}

AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t stream,
              const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  const size_t bytes = sizeof(data[0][0][0]) * STENCIL_DEPTH * STENCIL_HEIGHT *
                       STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyToSymbolAsync(
      stencils, data, bytes, stencil * bytes, cudaMemcpyHostToDevice, stream);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};

AcResult
acStoreStencil(const Stencil stencil, const cudaStream_t stream,
               AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
  ERRCHK_ALWAYS(stencil < NUM_STENCILS);

  const size_t bytes = sizeof(data[0][0][0]) * STENCIL_DEPTH * STENCIL_HEIGHT *
                       STENCIL_WIDTH;
  const cudaError_t retval = cudaMemcpyFromSymbolAsync(
      data, stencils, bytes, stencil * bytes, cudaMemcpyDeviceToHost, stream);

  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
};

#define GEN_LOAD_UNIFORM(LABEL_UPPER, LABEL_LOWER)                             \
  ERRCHK_ALWAYS(param < NUM_##LABEL_UPPER##_PARAMS);                           \
                                                                               \
  const size_t offset = (size_t)&d_mesh_info.LABEL_LOWER##_params[param] -     \
                        (size_t)&d_mesh_info;                                  \
                                                                               \
  const cudaError_t retval = cudaMemcpyToSymbolAsync(                          \
      d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice,      \
      stream);                                                                 \
  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;

AcResult
acLoadRealUniform(const cudaStream_t stream, const AcRealParam param,
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
acLoadReal3Uniform(const cudaStream_t stream, const AcReal3Param param,
                   const AcReal3 value)
{
  if (isnan(value.x) | isnan(value.y) | isnan(value.z)) {
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
acLoadIntUniform(const cudaStream_t stream, const AcIntParam param,
                 const int value)
{
  GEN_LOAD_UNIFORM(INT, int);
}

AcResult
acLoadInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                  const int3 value)
{
  GEN_LOAD_UNIFORM(INT3, int3);
}

#define GEN_STORE_UNIFORM(LABEL_UPPER, LABEL_LOWER)                            \
  ERRCHK_ALWAYS(param < NUM_##LABEL_UPPER##_PARAMS);                           \
                                                                               \
  const size_t offset = (size_t)&d_mesh_info.LABEL_LOWER##_params[param] -     \
                        (size_t)&d_mesh_info;                                  \
                                                                               \
  const cudaError_t retval = cudaMemcpyFromSymbolAsync(                        \
      value, d_mesh_info, sizeof(*value), offset, cudaMemcpyDeviceToHost,      \
      stream);                                                                 \
  return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;

AcResult
acStoreRealUniform(const cudaStream_t stream, const AcRealParam param,
                   AcReal* value)
{
  GEN_STORE_UNIFORM(REAL, real);
}

AcResult
acStoreReal3Uniform(const cudaStream_t stream, const AcReal3Param param,
                    AcReal3* value)
{
  GEN_STORE_UNIFORM(REAL3, real3);
}

AcResult
acStoreIntUniform(const cudaStream_t stream, const AcIntParam param, int* value)
{
  GEN_STORE_UNIFORM(INT, int);
}

AcResult
acStoreInt3Uniform(const cudaStream_t stream, const AcInt3Param param,
                   int3* value)
{
  GEN_STORE_UNIFORM(INT3, int3);
}

static TBConfig
autotune(const Kernel kernel, const int3 dims, VertexBufferArray vba)
{
  printf("Autotuning kernel %p, block (%d, %d, %d)... ", kernel, dims.x, dims.y,
         dims.z);
  fflush(stdout);

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
  const int warp_size             = props.warpSize;
  const int max_threads_per_block = props.maxThreadsPerBlock;

  for (int z = 1; z <= max_threads_per_block; ++z) {
    for (int y = 1; y <= max_threads_per_block; ++y) {
      for (int x = 1; x <= max_threads_per_block; ++x) {

        if (x * y * z > max_threads_per_block)
          break;

        // if (x * y * z * max_regs_per_thread > max_regs_per_block)
        //  break;

        if ((x * y * z) % warp_size)
          continue;

        // if (max_regs_per_block / (x * y * z) < min_regs_per_thread)
        //   continue;

        // if (x < y || x < z)
        //   continue;
        if (best_time < 100)
          break;

        const dim3 tpb(x, y, z);
        const dim3 bpg((unsigned int)ceil(dims.x / double(tpb.x)), //
                       (unsigned int)ceil(dims.y / double(tpb.y)), //
                       (unsigned int)ceil(dims.z / double(tpb.z)));
#if USE_SMEM
        const size_t smem = (tpb.x + STENCIL_WIDTH - 1 + pad) *
                            (tpb.y + STENCIL_HEIGHT - 1) * sizeof(AcReal);
#else
        const size_t smem = 0;
#endif

        cudaEvent_t tstart, tstop;
        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < num_iters; ++i)
          kernel<<<bpg, tpb, smem>>>(start, end, vba);
        cudaEventRecord(tstop); // Timing stop
        cudaEventSynchronize(tstop);

        if (cudaGetLastError() != cudaSuccess) // Discard failed runs
          continue;

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, tstart, tstop);

        if (milliseconds < best_time) {
          best_time = milliseconds;
          best_tpb  = tpb;
        }

        // printf("Auto-optimizing... Current tpb: (%d, %d, %d), time %f ms\n",
        //       tpb.x, tpb.y, tpb.z, (double)best_time / num_iters);
        fflush(stdout);
      }
    }
  }
  c.tpb = best_tpb;

  printf("The best tpb: (%d, %d, %d), time %f ms\n", best_tpb.x, best_tpb.y,
         best_tpb.z, (double)best_time / num_iters);

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
