/*
 * Random number generation
 */
#if AC_USE_HIP
#include <hip/hip_runtime.h> // Needed in files that include kernels

#include <hip/hip_fp16.h>           // Workaround: required by hiprand
#include <hiprand/hiprand.h>        // Random numbers
#include <hiprand/hiprand_kernel.h> // Random numbers (device)
#else
#include <curand.h>        // Random numbers
#include <curand_kernel.h> // Random numbers (device)
#endif

typedef curandStateXORWOW_t acRandState;
static __device__ __constant__ acRandState* rand_states;

__global__ void
rand_init(const uint64_t seed, const size_t count, const size_t rank)
{
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count)
    return;

  const size_t gtid = tid + rank * count;
  curand_init(seed, gtid, 0, &rand_states[tid]);
}

AcResult
acRandInitAlt(const uint64_t seed, const size_t count, const size_t rank)
{
  acRandState* h_rand_states = NULL;
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&h_rand_states, count * sizeof(acRandState)));
  ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(rand_states,&h_rand_states, sizeof(h_rand_states), 0, cudaMemcpyHostToDevice));


  const size_t tpb = 256;
  const size_t bpg = as_size_t(ceil(1. * count / tpb));
  rand_init<<<bpg, tpb>>>(seed, count, rank);
  ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());

  return AC_SUCCESS;
}

void
acRandQuit(void)
{
  ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
  acRandState* h_rand_states = NULL;
  ERRCHK_CUDA_ALWAYS(cudaMemcpyFromSymbol(&h_rand_states, rand_states, sizeof(h_rand_states), 0, cudaMemcpyDeviceToHost));
  ERRCHK_CUDA_ALWAYS(cudaFree(h_rand_states));
  h_rand_states = NULL;
  ERRCHK_CUDA_ALWAYS(cudaMemcpyToSymbol(rand_states,&h_rand_states, sizeof(h_rand_states), 0, cudaMemcpyHostToDevice));

}

/*
// Not working
__global__ void
rand_init(const uint64_t seed, const Volume m_local, const Volume m_global,
          const Volume global_offset)
{
  const Volume tid = (Volume){
      threadIdx.x * blockIdx.x * blockDim.x,
      threadIdx.y * blockIdx.y * blockDim.y,
      threadIdx.z * blockIdx.z * blockDim.z,
  };
  if (tid.x >= m_local.x)
    return;
  if (tid.y >= m_local.y)
    return;
  if (tid.z >= m_local.z)
    return;

  const size_t local_idx = tid.x               //
                           + tid.y * m_local.x //
                           + tid.z * m_local.x * m_local.y;

  const Volume gtid = (Volume){
      tid.x + global_offset.x,
      tid.y + global_offset.y,
      tid.z + global_offset.z,
  };
  const size_t global_idx = gtid.x                //
                            + gtid.y * m_global.x //
                            + gtid.z * m_global.x * m_global.y;

  curand_init(seed, global_idx, 0, &rand_states[local_idx]);
}

AcResult
acRandInit(const uint64_t seed, const Volume m_local, const Volume m_global,
           const Volume global_offset)
{
  const Volume tpb   = (Volume){256, 4, 1};
  const Volume bpg   = get_bpg(m_local, tpb);
  const size_t count = m_local.x * m_local.y * m_local.z;
  // const size_t count = (tpb.x * bpg.x) * (tpb.y * bpg.y) * (tpb.z * bpg.z);
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&rand_states, count * sizeof(rand_states[0])));
  ERRCHK_ALWAYS(rand_states);

  rand_init<<<to_dim3(bpg), to_dim3(tpb)>>>(seed, m_local, m_global,
                                            global_offset);
  ERRCHK_CUDA_KERNEL_ALWAYS();
  cudaDeviceSynchronize();

  return AC_SUCCESS;
}*/

#if 0
// WORKING
/*
__global__ void
rand_init(const uint64_t seed, const Volume m_local, const Volume m_global,
          const Volume global_offset)
{
  const Volume tid = (Volume){
      threadIdx.x * blockIdx.x * blockDim.x,
      threadIdx.y * blockIdx.y * blockDim.y,
      threadIdx.z * blockIdx.z * blockDim.z,
  };
  if (tid.x >= m_local.x)
    return;
  if (tid.y >= m_local.y)
    return;
  if (tid.z >= m_local.z)
    return;

  const size_t local_idx = tid.x               //
                           + tid.y * m_local.x //
                           + tid.z * m_local.x * m_local.y;

  const Volume gtid = (Volume){
      tid.x + global_offset.x,
      tid.y + global_offset.y,
      tid.z + global_offset.z,
  };
  const size_t global_idx = gtid.x                //
                            + gtid.y * m_global.x //
                            + gtid.z * m_global.x * m_global.y;
  curand_init(seed, local_idx, 0, &rand_states[local_idx]);
}
*/

__global__ void
rand_init(const uint64_t seed, const size_t count)
{
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < count)
    curand_init(seed, tid, 0, &rand_states[tid]);
}

AcResult
acRandInit(const uint64_t seed, const Volume m_local, const Volume m_global,
           const Volume global_offset)
{
  /*
  const Volume tpb   = (Volume){128, 4, 2};
  const Volume bpg   = get_bpg(m_local, tpb);
  const size_t count = tpb.x * bpg.x * tpb.y * bpg.y * tpb.z * bpg.z;
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&rand_states, count * sizeof(rand_states[0])));
  ERRCHK_ALWAYS(rand_states);

  rand_init<<<to_dim3(bpg), to_dim3(tpb)>>>(seed, m_local, m_global,
                                            global_offset);
  cudaDeviceSynchronize();
  ERRCHK_CUDA_KERNEL_ALWAYS();
  */
  const size_t count = m_local.x * m_local.y * m_local.z;
  ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&rand_states, count * sizeof(rand_states[0])));

  const size_t tpb = 1024;
  const size_t bpg = ceil(1. * count / tpb);

  rand_init<<<bpg, tpb>>>(seed, count);
  cudaDeviceSynchronize();

  return AC_SUCCESS;
}
#endif


#if AC_DOUBLE_PRECISION
#define rand_uniform() curand_uniform_double(&rand_states[local_compdomain_idx])
#else
#define rand_uniform() curand_uniform(&rand_states[local_compdomain_idx])
#endif
__device__ __forceinline__
AcReal
random_uniform(const size_t idx)
{
#if AC_DOUBLE_PRECISION
		return curand_uniform_double(&rand_states[idx]);
#else
		return curand_uniform(&rand_states[idx]);
#endif
}
