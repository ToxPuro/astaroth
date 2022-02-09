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
#include <stdio.h>

#if AC_USE_HIP
#include "hip.h"
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime_api.h> // cudaStream_t
#endif

#include "datatypes.h"
#include "errchk.h"

#include "user_defines.h"

typedef struct {
  int int_params[NUM_INT_PARAMS];
  int3 int3_params[NUM_INT3_PARAMS];
  AcReal real_params[NUM_REAL_PARAMS];
  AcReal3 real3_params[NUM_REAL3_PARAMS];
} AcMeshInfo;

typedef struct {
  AcReal* in[NUM_FIELDS];
  AcReal* out[NUM_FIELDS];
} VertexBufferArray;

typedef void (*Kernel)(const int3, const int3, VertexBufferArray vba);

#ifdef __cplusplus
extern "C" {
#endif

#include "user_declarations.h"

AcResult acLaunchKernel(Kernel func, const cudaStream_t stream,
                        const int3 start, const int3 end,
                        VertexBufferArray vba);

AcResult
acLoadStencil(const Stencil stencil, const cudaStream_t stream,
              const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

AcResult
acStoreStencil(const Stencil stencil, const cudaStream_t stream,
               AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]);

#define GEN_LOAD_UNIFORM_DECLARATION(TYPE)                                     \
  AcResult acLoad##TYPE##Uniform(const cudaStream_t stream, TYPE symbol,       \
                                 const TYPE value)

#define GEN_STORE_UNIFORM_DECLARATION(TYPE)                                    \
  AcResult acStore##TYPE##Uniform(const cudaStream_t stream, TYPE* dst,        \
                                  const TYPE symbol)

GEN_LOAD_UNIFORM_DECLARATION(AcReal);
GEN_LOAD_UNIFORM_DECLARATION(AcReal3);
GEN_LOAD_UNIFORM_DECLARATION(int);
GEN_LOAD_UNIFORM_DECLARATION(int3);

GEN_STORE_UNIFORM_DECLARATION(AcReal);
GEN_STORE_UNIFORM_DECLARATION(AcReal3);
GEN_STORE_UNIFORM_DECLARATION(int);
GEN_STORE_UNIFORM_DECLARATION(int3);

#ifdef __cplusplus
} // extern "C"
#endif
