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
#include "acc_runtime.h"
#include <stdbool.h>

typedef AcReal AcRealPacked;

#ifdef __cplusplus
extern "C" {
#endif
AcResult acKernelDummy(void);

typedef struct AcShearInterpolationCoeffs
{
	AcReal c1,c2,c3,c4,c5,c6;
} AcShearInterpolationCoeffs;

#include "common_kernels.h"

AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const Volume vba_start, const Volume dims, AcRealPacked* packed,
			  const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const Volume vba_start, const Volume dims, VertexBufferArray vba,
			    const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

AcResult
acKernelShearUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const Volume vba_start, const Volume dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs,
                          const AcShearInterpolationCoeffs coeffs, const int offset
                          );

AcResult
acKernelMoveData(const cudaStream_t stream, const Volume src_start, const Volume dst_start, const Volume src_dims, const Volume dst_dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs);

/** */
size_t acKernelReduceGetMinimumScratchpadSize(const int3 max_dims);

/** */
size_t acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const AcReduction reduction,
                          const VertexBufferHandle vtxbuf, const Volume start, const Volume end,
			  const int scratchpad_index,
                          VertexBufferArray vba);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                         const Volume end, const Field3 vec, VertexBufferArray vba,
			 const int scratchpad_index
                         );

/** */
AcReal
acKernelReduceVecScal(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                      const Volume end, const Field4 vtxbufs,VertexBufferArray vba,
		      const int scratchpad_index
                   );

/** */
AcResult acKernelVolumeCopy(const cudaStream_t stream,                                    //
                            const AcReal* in, const Volume in_offset, const Volume in_volume, //
                            AcReal* out, const Volume out_offset, const Volume out_volume);

AcResult acReduceReal(const cudaStream_t stream, const AcReduceOp, AcRealScalarReduceBuffer, const size_t count);

AcResult acReduceInt(const cudaStream_t stream, const AcReduceOp, AcIntScalarReduceBuffer, const size_t count);

#if AC_DOUBLE_PRECISION
AcResult acReduceFloat(const cudaStream_t stream, const AcReduceOp, AcFloatScalarReduceBuffer, const size_t count);
#endif

AcResult acSegmentedReduce(const cudaStream_t stream, const AcReal* d_in,
                           const size_t count, const size_t num_segments,
                           AcReal* d_out, AcReal** tmp, size_t* tmp_size);
AcResult
acReduceProfileWithBounds(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream, const Volume start, const Volume end, const Volume start_after_transpose, const Volume end_after_transpose);

AcResult
acReduceProfile(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream);

#include "transpose.h"

AcResult
acReduceClean();

AcResult
acKernelsClean();

// Astaroth 2.0 backwards compatibility.

#ifdef __cplusplus
} // extern "C"

// cplusplus overloads
//

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const Volume vba_start, const Volume dims, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const Volume vba_start, const Volume dims, AcRealPacked* packed);

static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcRealScalarReduceBuffer buffer, const size_t count)
{
	return acReduceReal(stream,op,buffer,count);
}

static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcIntScalarReduceBuffer buffer, const size_t count)
{
	return acReduceInt(stream,op,buffer,count);
}

#if AC_DOUBLE_PRECISION
static UNUSED AcResult
acReduce(const cudaStream_t stream, const AcReduceOp op, AcFloatScalarReduceBuffer buffer, const size_t count)
{
	return acReduceFloat(stream,op,buffer,count);
}
#endif
#endif

template <int direction>  static __global__ void packUnpackPlate(AcReal* __restrict__ buffer, VertexBufferArray vba, int3 start, int3 end);


