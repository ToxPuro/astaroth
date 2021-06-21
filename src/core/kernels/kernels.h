#pragma once
#include "astaroth.h"

#if AC_MPI_ENABLED
#include <mpi.h>
#include <stdbool.h>

#define MPI_GPUDIRECT_DISABLED (0)
#endif // AC_MPI_ENABLED

typedef AcReal AcRealPacked;

typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];

    AcReal* profiles[NUM_SCALARARRAY_HANDLES];
} VertexBufferArray;

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

typedef struct {
    cudaStream_t stream;
    int step_number;
    int3 start;
    int3 end;
} KernelParameters;

typedef AcResult (*ComputeKernel)(const KernelParameters, VertexBufferArray);

#ifdef __cplusplus
extern "C" {
#endif

/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);

/** */
AcResult acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 normal, const int3 dims,
                                    AcReal* vtxbuf);


/** */
AcResult acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                   AcReal* vtxbuf, const VertexBufferHandle vtxbuf_handle,
                                   const AcMeshInfo config, const int3 bindex);

/** */
AcResult acKernelDummy(void);

/** */
AcResult acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, const int3 dims, AcRealPacked* packed);

/** */
AcResult acKernelIntegrateSubstep(const KernelParameters params, VertexBufferArray vba);

/** */
AcResult acKernelPartialPackData(const cudaStream_t stream, const VertexBufferArray vba,
                                 const int3 vba_start, const int3 dims, AcRealPacked* packed,
                                 VertexBufferHandle variable_scope[], size_t var_scope_len);

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const int3 vba_start, const int3 dims, VertexBufferArray vba);

/** */
AcResult acKernelPartialUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                                   const int3 vba_start, const int3 dims, VertexBufferArray vba,
                                   VertexBufferHandle variable_scope[], size_t var_scope_len);

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

#define GEN_KERNEL_FUNC_DECL(ID)                                                                   \
    AcResult AC_KERNEL_FUNC_NAME(ID)(const KernelParameters params, VertexBufferArray vba);

#include "user_kernel_decl.h"

#ifdef __cplusplus
} // extern "C"
#endif
