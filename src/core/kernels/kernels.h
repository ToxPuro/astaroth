#pragma once
#include "astaroth.h"

#if AC_MPI_ENABLED
#include <mpi.h>
#include <stdbool.h>

#define MPI_GPUDIRECT_DISABLED (0)
#endif // AC_MPI_ENABLED

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;
};

typedef AcReal AcRealPacked;

typedef struct {
    Kernel kernel;
    cudaStream_t stream;
    int step_number;
    int3 start;
    int3 end;
} KernelParameters;

#ifdef __cplusplus
extern "C" {
#endif

/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);

/** */
AcResult acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 region_id,
                                     const int3 normal, const int3 dims, AcReal* vtxbuf);

/** */
AcResult acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                   AcReal* vtxbuf, const VertexBufferHandle vtxbuf_handle,
                                   const AcMeshInfo config, const int3 bindex);

/** */
AcResult acKernelDummy(void);

/** */
// AcResult acKernelAutoOptimizeIntegration(const int3 start, const int3 end,
// VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, const int3 dims, AcRealPacked* packed);

/** */
// AcResult acKernelIntegrateSubstep(const KernelParameters params,
// VertexBufferArray vba);

/** */
AcResult acKernelPartialPackData(const cudaStream_t stream, const VertexBufferArray vba,
                                 const int3 vba_start, const int3 dims, AcRealPacked* packed,
                                 VertexBufferHandle vtxbufs[], size_t num_vtxbufs);

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const int3 vba_start, const int3 dims, VertexBufferArray vba);

/** */
AcResult acKernelPartialUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                                   const int3 vba_start, const int3 dims, VertexBufferArray vba,
                                   VertexBufferHandle vtxbufs[], size_t num_vtxbufs);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                          const int3 end, const AcReal* vtxbuf, AcReal* scratchpad,
                          AcReal* reduce_result);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                         const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                         const AcReal* vtxbuf2, AcReal* scratchpad, AcReal* reduce_result);

/** */
AcReal acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                             const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                             const AcReal* vtxbuf2, const AcReal* vtxbuf3, AcReal* scratchpad,
                             AcReal* reduce_result);

// Astaroth 2.0 backwards compatibility.
AcResult acKernel(const KernelParameters params, VertexBufferArray vba);

#ifdef __cplusplus
} // extern "C"
#endif