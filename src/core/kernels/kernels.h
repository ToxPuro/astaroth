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

// Generic boundconds

/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);

/** */
AcResult acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 region_id,
                                     const int3 normal, const int3 dims, AcReal* vtxbuf);

/** */
AcResult
acKernelA2Boundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                     const int3 dims, AcReal* vtxbuf);

/** */
AcResult
acKernelConstantDerivativeBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                                     const int3 dims, AcReal* vtxbuf);

// Entropy boundconds

/** */
AcResult acKernelEntropyConstantTemperatureBoundconds(const cudaStream_t stream, const int3 region_id,
                                                      const int3 normal, const int3 dims, VertexBufferArray vba);

/** */
AcResult acKernelEntropyBlackbodyRadiationKramerConductivityBoundconds(const cudaStream_t stream, const int3 region_id,
                                                                       const int3 normal, const int3 dims, VertexBufferArray vba);

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
