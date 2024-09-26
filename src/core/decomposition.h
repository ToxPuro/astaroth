/*
    Copyright (C) 2014-2024, Johannes Pekkila, Miikka Vaisala.

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
#include <stdint.h> //uint64_t

#include "errchk.h"
#include "math_utils.h" //uint64_t, uint3_64
#include "astaroth.h"

typedef struct {
    size_t ndims;
    size_t* global_dims; // [ndims]
    size_t* local_dims;  // [ndims]

    size_t nlayers;
    size_t* decomposition;        // [nlayers*ndims]
    size_t* global_decomposition; // [ndims]
} AcDecompositionInfo;

void acDecompositionInfoPrint(const AcDecompositionInfo info);

AcDecompositionInfo acDecompositionInfoCreate(const size_t ndims, const size_t* global_dims,
                                              const size_t nlayers,
                                              const size_t* partitions_per_layer);

void acDecompositionInfoDestroy(AcDecompositionInfo* info);

int acGetPid(const int3 pid_input, const AcDecompositionInfo info);

int3 acGetPid3D(const int i, const AcDecompositionInfo info);

// --------------------
// Backwards compatibility
// --------------------
void compat_acDecompositionInit(const size_t ndims, const size_t* global_dims, const size_t nlayers,
                                const size_t* partitions_per_layer);
void compat_acDecompositionQuit(void);

uint3_64 decompose(const uint64_t target, const AcDecomposeStrategy decompose_strategy);

int getPid(const int3 pid_raw, const uint3_64 decomp,const int proc_mapping_strategy);

int3 getPid3D(const uint64_t pid, const uint3_64 decomp,const int proc_mapping_strategy);
