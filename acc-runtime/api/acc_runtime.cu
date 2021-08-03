#include "acc_runtime.h"

#include <vector> // tbconfig

#include "errchk.h"
#include "math_utils.h"

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

__device__ AcMeshInfo d_mesh_info;

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
#define real3(i, j, k) ((AcReal3){(i), (j), (k)})
#define print printf // TODO is this a good idea?

#include "user_kernels.h"

typedef struct {
  Kernel kernel;
  int3 dims;
  dim3 tpb;
} TBConfig;

static std::vector<TBConfig> tbconfigs;

static TBConfig getOptimalTBConfig(const Kernel kernel, const int3 dims,
                                   VertexBufferArray vba);

AcResult
acLaunchKernel(Kernel kernel, const cudaStream_t stream, const int3 start,
               const int3 end, VertexBufferArray vba)
{
  const int3 n = end - start;

  const dim3 tpb = getOptimalTBConfig(kernel, n, vba).tpb;
  const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                 (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                 (unsigned int)ceil(n.z / AcReal(tpb.z)));
  const size_t smem = 0;

  kernel<<<bpg, tpb, smem, stream>>>(start, end, vba);
  ERRCHK_CUDA_KERNEL();

  return AC_SUCCESS;
}

#define GEN_LOAD_UNIFORM(TYPE)                                                 \
  GEN_LOAD_UNIFORM_DECLARATION(TYPE)                                           \
  {                                                                            \
    cudaError_t retval = cudaMemcpyToSymbolAsync(                              \
        symbol, &value, sizeof(value), 0, cudaMemcpyHostToDevice, stream);     \
    return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;                    \
  }

#define GEN_STORE_UNIFORM(TYPE)                                                \
  GEN_STORE_UNIFORM_DECLARATION(TYPE)                                          \
  {                                                                            \
    cudaError_t retval = cudaMemcpyFromSymbolAsync(                            \
        dst, symbol, sizeof(*dst), 0, cudaMemcpyDeviceToHost, stream);         \
    return retval == cudaSuccess ? AC_SUCCESS : AC_FAILURE;                    \
  }

GEN_LOAD_UNIFORM(AcReal)
GEN_LOAD_UNIFORM(AcReal3)
GEN_LOAD_UNIFORM(int)
GEN_LOAD_UNIFORM(int3)

GEN_STORE_UNIFORM(AcReal)
GEN_STORE_UNIFORM(AcReal3)
GEN_STORE_UNIFORM(int)
GEN_STORE_UNIFORM(int3)

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
  const int num_iters = 10;

  for (int z = 1; z <= MAX_THREADS_PER_BLOCK; ++z) {
    for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
      for (int x = 1; x <= MAX_THREADS_PER_BLOCK; ++x) {

        if (x * y * z > MAX_THREADS_PER_BLOCK)
          break;

        if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
          break;

        const dim3 tpb(x, y, z);
        const dim3 bpg((unsigned int)ceil(dims.x / AcReal(tpb.x)), //
                       (unsigned int)ceil(dims.y / AcReal(tpb.y)), //
                       (unsigned int)ceil(dims.z / AcReal(tpb.z)));

        cudaEvent_t tstart, tstop;
        cudaEventCreate(&tstart);
        cudaEventCreate(&tstop);

        cudaDeviceSynchronize();
        cudaEventRecord(tstart); // Timing start
        for (int i = 0; i < num_iters; ++i)
          kernel<<<bpg, tpb>>>(start, end, vba);
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
/*
static TBConfig
autotune(const int3 dims, VertexBufferArray vba)
{
    fprintf(stderr, "------------------TODO WARNING FIX autotune
HARMFUL!----------------\ndt not " "set properly and MUST call w/ all possible
subdomain sizes before actual " "simulation with dt = 0, otherwise advances the
simulation arbitrarily!!\n");

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + dims;

    printf("Autotuning for (%d, %d, %d)... ", dims.x, dims.y, dims.z);
    // RK3
    dim3 best_dims(0, 0, 0);
    float best_time          = INFINITY;
    const int num_iterations = 10;

    for (int z = 1; z <= MAX_THREADS_PER_BLOCK; ++z) {
        for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
            for (int x = 4; x <= MAX_THREADS_PER_BLOCK; x += 4) {

                // if (x > end.x - start.x || y > end.y - start.y || z > end.z -
start.z)
                //    break;

                if (x * y * z > MAX_THREADS_PER_BLOCK)
                    break;

                if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
                    break;

                // if (((x * y * z) % WARP_SIZE) != 0)
                //    continue;

                const dim3 tpb(x, y, z);
                const int3 n = end - start;
                const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                               (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                               (unsigned int)ceil(n.z / AcReal(tpb.z)));
                const size_t smem = 0;

                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) // resets the error if
any continue;

                // printf("(%d, %d, %d)\n", x, y, z);

                cudaEvent_t tstart, tstop;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);

                cudaEventRecord(tstart); //
---------------------------------------- Timing start for (int i = 0; i <
num_iterations; ++i) solve<2><<<bpg, tpb, smem, 0>>>(start, end, vba);

                cudaEventRecord(tstop); //
----------------------------------------- Timing end
                cudaEventSynchronize(tstop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, tstart, tstop);

                ERRCHK_CUDA_KERNEL_ALWAYS();
                // printf("(%d, %d, %d): %.4g ms\n", x, y, z,
(double)milliseconds /
                // num_iterations); fflush(stdout);
                if (milliseconds < best_time) {
                    best_time = milliseconds;
                    best_dims = tpb;
                }
            }
        }
    }
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);
    //#if AC_VERBOSE
    printf("Auto-optimization done. The best threadblock dimensions for rkStep
(%d, %d, %d): "
           "(%d, "
           "%d, %d) "
           "%f "
           "ms\n",
           dims.x, dims.y, dims.z, best_dims.x, best_dims.y, best_dims.z,
           double(best_time) / num_iterations);
    //#endif

    // Failed to find valid thread block dimensions
    ERRCHK_ALWAYS(best_dims.x * best_dims.y * best_dims.z > 0);

    TBConfig c;
    c.tpb  = best_dims;
    c.dims = dims;
    return c;
}
*/

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