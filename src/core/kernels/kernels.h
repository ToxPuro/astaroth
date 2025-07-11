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

AcResult
acComplexToReal(const AcComplex* src, const size_t count, AcReal* dst);

AcResult
acRealToComplex(const AcReal* src, const size_t count, AcComplex* dst);

AcResult
acPlanarToComplex(const AcReal* real_src, const AcReal* imag_src,const size_t count, AcComplex* dst);

AcResult
acComplexToPlanar(const AcComplex* src,const size_t count,AcReal* real_dst,AcReal* imag_dst);

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

// Astaroth 2.0 backwards compatibility.

#ifdef __cplusplus
} // extern "C"

// cplusplus overloads
//
AcResult
acKernelFlush(const cudaStream_t stream, AcReal* arr, const size_t n,
              const AcReal value);

AcResult
acKernelFlush(const cudaStream_t stream, int* arr, const size_t n,
              const int value);
AcResult
acKernelFlush(const cudaStream_t stream, AcComplex* arr, const size_t n,
              const AcComplex value);

#if AC_DOUBLE_PRECISION
AcResult
acKernelFlush(const cudaStream_t stream, float* arr, const size_t n,
              const float value);
#endif

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                            const Volume vba_start, const Volume dims, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const Volume vba_start, const Volume dims, AcRealPacked* packed);
#endif

template <int direction>  static __global__ void packUnpackPlate(AcReal* __restrict__ buffer, VertexBufferArray vba, int3 start, int3 end);


