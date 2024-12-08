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
//#include "astaroth.h"
#include "acc_runtime.h"

#include <stdbool.h>
#if AC_MPI_ENABLED
//#include <mpi.h>

#define MPI_GPUDIRECT_DISABLED (0)
#endif // AC_MPI_ENABLED

typedef struct
{
	AcReduceOp reals[NUM_REAL_SCRATCHPADS+1];	
	AcReduceOp ints[NUM_INT_OUTPUTS+1];	
} AcScratchpadStates;


typedef AcReal AcRealPacked;


#ifdef __cplusplus
extern "C" {
#endif


//TP: deprecated
//AcResult acKernelGeneralBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
//                                   AcReal* vtxbuf, const VertexBufferHandle vtxbuf_handle,
//                                   const AcMeshInfoParams config, const int3 bindex);
//
/** */
AcResult acKernelDummy(void);

/** */
// AcResult acKernelAutoOptimizeIntegration(const int3 start, const int3 end,
// VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, const int3 dims, AcRealPacked* packed,
			  const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const int3 vba_start, const int3 dims, VertexBufferArray vba,
			    const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

AcResult
acKernelMoveData(const cudaStream_t stream, const int3 src_start, const int3 dst_start, const int3 src_dims, const int3 dst_dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

/** */
// AcResult acKernelIntegrateSubstep(const KernelParameters params,
// VertexBufferArray vba);

/** */

/** */
size_t acKernelReduceGetMinimumScratchpadSize(const int3 max_dims);

/** */
size_t acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const AcReduction reduction,
                          const VertexBufferHandle vtxbuf, const int3 start, const int3 end,
                          AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                          const size_t scratchpad_size, VertexBufferArray vba);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const AcReduction reduction, const int3 start,
                         const int3 end, const Field3 vec, VertexBufferArray vba,
                         AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS],
                         const size_t scratchpad_size);

/** */
AcReal
acKernelReduceVecScal(const cudaStream_t stream, const AcReduction reduction, const int3 start,
                      const int3 end, const Field4 vtxbufs,VertexBufferArray vba,
                      AcReal* scratchpads[NUM_REDUCE_SCRATCHPADS], const size_t scratchpad_size);

/** */
AcResult acKernelVolumeCopy(const cudaStream_t stream,                                    //
                            const AcReal* in, const int3 in_offset, const int3 in_volume, //
                            AcReal* out, const int3 out_offset, const int3 out_volume);

// Astaroth 2.0 backwards compatibility.
//
//TP: not used and should anyway be behind macro defs
//void acUnpackPlate(const Device device, int3 start, int3 end, int block_size, const Stream stream, int plate);
//void acPackPlate(const Device device, int3 start, int3 end, int block_size, const Stream stream, int plate);

#ifdef __cplusplus
} // extern "C"

// cplusplus overloads

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const int3 vba_start, const int3 dims, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, const int3 dims, AcRealPacked* packed);
#endif

template <int direction>  static __global__ void packUnpackPlate(AcReal* __restrict__ buffer, VertexBufferArray vba, int3 start, int3 end);


