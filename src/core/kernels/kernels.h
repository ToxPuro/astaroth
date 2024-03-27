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
    AcReal* reduce_scratchpads[NUM_REDUCE_SCRATCHPADS];
    size_t scratchpad_size;

    // LTFM
    AcBufferArray tfm_scratchpads;
    // LTFM
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

// Generic boundconds

/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);

/** */
AcResult acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 region_id,
                                     const int3 normal, const int3 dims, AcReal* vtxbuf);

/** */
AcResult acKernelAntiSymmetricBoundconds(const cudaStream_t stream, const int3 region_id,
                                         const int3 normal, const int3 dims, AcReal* vtxbuf);

/** */
AcResult acKernelA2Boundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                              const int3 dims, AcReal* vtxbuf);

/** */
AcResult acKernelConstBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                                 const int3 dims, AcReal* vtxbuf, AcRealParam const_value);

/** */
AcResult acKernelPrescribedDerivativeBoundconds(const cudaStream_t stream, const int3 region_id,
                                                const int3 normal, const int3 dims, AcReal* vtxbuf,
                                                AcRealParam der_val_param);
/** */
AcResult acKernelOutflowBoundconds(const cudaStream_t stream, const int3 region_id,
                                   const int3 normal, const int3 dims, AcReal* vtxbuf);
/** */
AcResult acKernelInflowBoundconds(const cudaStream_t stream, const int3 region_id,
                                  const int3 normal, const int3 dims, AcReal* vtxbuf);

// Entropy boundconds

#ifdef AC_INTEGRATION_ENABLED
/** */
AcResult acKernelEntropyConstantTemperatureBoundconds(const cudaStream_t stream,
                                                      const int3 region_id, const int3 normal,
                                                      const int3 dims, VertexBufferArray vba);

/** */
AcResult acKernelEntropyBlackbodyRadiationKramerConductivityBoundconds(const cudaStream_t stream,
                                                                       const int3 region_id,
                                                                       const int3 normal,
                                                                       const int3 dims,
                                                                       VertexBufferArray vba);

/** */
AcResult acKernelEntropyPrescribedHeatFluxBoundconds(const cudaStream_t stream,
                                                     const int3 region_id, const int3 normal,
                                                     const int3 dims, VertexBufferArray vba,
                                                     AcRealParam F_param);

AcResult acKernelEntropyPrescribedNormalAndTurbulentHeatFluxBoundconds(
    const cudaStream_t stream, const int3 region_id, const int3 normal, const int3 dims,
    VertexBufferArray vba, AcRealParam hcond_param, AcRealParam F_param);

#endif

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
size_t acKernelReduceGetMinimumScratchpadSize(const int3 max_dims);

/** */
size_t acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype,
                          const AcReal* vtxbuf, const int3 start, const int3 end,
                          AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                          const size_t scratchpad_size);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                         const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                         const AcReal* vtxbuf2, AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                         const size_t scratchpad_size);

/** */
AcReal acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                             const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                             const AcReal* vtxbuf2, const AcReal* vtxbuf3,
                             AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                             const size_t scratchpad_size);

/** */
AcResult acKernelVolumeCopy(const cudaStream_t stream,                                    //
                            const AcReal* in, const int3 in_offset, const int3 in_volume, //
                            AcReal* out, const int3 out_offset, const int3 out_volume);

/** */
AcReal acKernelReduceScalAsyncToOutput(const cudaStream_t stream, const ReductionType rtype,
                                       const AcReal* vtxbuf, const int3 start, const int3 end,
                                       AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                                       const size_t scratchpad_size, AcReal* output);

// Astaroth 2.0 backwards compatibility.
AcResult acKernel(const KernelParameters params, VertexBufferArray vba);

#ifdef __cplusplus
} // extern "C"
#endif
