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
#include <stddef.h> // size_t
#include <stdint.h> //uint64_t

typedef struct {
    size_t ndims;
    size_t* global_dims; // [ndims]
    size_t* local_dims;  // [ndims]

    size_t nlayers;
    size_t* decomposition;        // [nlayers*ndims]
    size_t* global_decomposition; // [ndims]
} AcDecompositionInfo;

void acDecompositionInfoPrint(const AcDecompositionInfo info);

/**
 * global_dims: global mesh dimensions nn
 * partitions_per_layer: partitioning hierarchy from highest granularity to lowest
 */
AcDecompositionInfo acDecompositionInfoCreate(const size_t ndims, const size_t* global_dims,
                                              const size_t nlayers,
                                              const size_t* partitions_per_layer);

void acDecompositionInfoDestroy(AcDecompositionInfo* info);

size_t acGetPid(const size_t ndims, const int64_t* pid_input, const AcDecompositionInfo info);

void acGetPid3D(const size_t i, const AcDecompositionInfo info, const size_t ndims,
                int64_t* pid_output);

// void acVerifyDecomposition(const AcDecompositionInfo info);
