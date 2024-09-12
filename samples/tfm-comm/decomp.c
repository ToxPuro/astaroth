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
#include "decomp.h"

#include <stdio.h>
#include <stdlib.h> // malloc
#include <string.h> // memcpy

#include "errchk.h"
#include "math_utils.h"
#include "print.h"
#include "type_conversion.h"

void
acDecompositionInfoPrint(const AcDecompositionInfo info)
{
    printf("AcDecompositionInfo %p\n", &info);
    acPrint_size_t("\tndims", info.ndims);
    acPrint_size_t("\tnlayers", info.nlayers);
    acPrintArray_size_t("\tglobal_dims", info.ndims, info.global_dims);
    acPrintArray_size_t("\tlocal_dims", info.ndims, info.local_dims);
    acPrintArray_size_t("\tdecomposition", info.ndims * info.nlayers, info.decomposition);
    acPrintArray_size_t("\tglobal_decomposition", info.ndims, info.global_decomposition);
}

static void
dims_create(const size_t nprocs, const size_t ndims, const size_t* dims, //
            size_t* local_dims, size_t* decomposition)
{
    // Do not allow aliasing
    ERRCHK(dims < local_dims || dims > local_dims + ndims);

    for (size_t i = 0; i < ndims; ++i) {
        local_dims[i]    = dims[i];
        decomposition[i] = 1;
    }
    if (nprocs == 1)
        return;

    size_t nfactors;
    factorize(nprocs, &nfactors, NULL);
    size_t* factors = (size_t*)malloc(sizeof(factors[0]) * nfactors);
    factorize(nprocs, &nfactors, factors);
    // print("factors", nfactors, factors);
    nfactors = unique(nfactors, factors);
    // print("unique", nfactors, factors);
    while (prod(ndims, decomposition) != nprocs) {
        // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
        // Adapted to try out all factors to work with a wider range of dims
        // Or maybe this is what they meant all along, but their description was just unclear
        size_t bestj         = SIZE_MAX;
        size_t bestlocal_dim = 0;
        size_t bestfac       = 0;
        for (size_t i = 0; i < nfactors; ++i) {
            const size_t fac = factors[nfactors - 1 - i];
            // printf("fac %zu\n", fac);
            // print("local dims", ndims, local_dims);
            for (size_t j = 0; j < ndims; ++j) {
                // if ((local_dims[j] % fac) == 0)
                //     printf("\t[%zu] = %zu divisible by %zu\n", j, local_dims[j], fac);
                if ((local_dims[j] % fac) == 0 && (local_dims[j] > bestlocal_dim)) {
                    bestj         = j;
                    bestlocal_dim = local_dims[j];
                    bestfac       = fac;
                }
            }
        }
        // printf("chose [%zu] = %zu\n", bestj, bestlocal_dim);
        ERRCHK(bestj < ndims); // Failed to find proper dimensions
        ERRCHK(prod(ndims, decomposition) <= nprocs);
        decomposition[bestj] *= bestfac;
        local_dims[bestj] /= bestfac;
    }
    free(factors);
    // printf("---COMPLETE--\n");

    ERRCHK(prod(ndims, local_dims) * prod(ndims, decomposition) == prod(ndims, dims));
    fflush(stdout);
}

static int
acHierarchicalDomainDecomposition(const size_t ndims, const size_t* dims,                   //
                                  const size_t nlayers, const size_t* partitions_per_layer, //
                                  size_t* local_dims, size_t* decompositions)
{
    // print("Partitions per layer", nlayers, partitions_per_layer);
    // print("Dimensions", ndims, dims);

    size_t global_dims[ndims];
    memcpy(global_dims, dims, ndims * sizeof(dims[0]));

    for (size_t j = nlayers - 1; j < nlayers; --j) {
        // size_t local_dims[ndims];
        dims_create(partitions_per_layer[j], ndims, global_dims, local_dims,
                    &decompositions[j * ndims]);
        memcpy(global_dims, local_dims, ndims * sizeof(dims[0]));
        // printf("\tLayer %zu\n", j);
        // print("\tGlobal dims", ndims, global_dims);
        // print("\tLocal dims", ndims, local_dims);
        // print("\tDecomposition", ndims, &decompositions[j * ndims]);
        // printf("\n");
    }
    // print("Decomposition", ndims * nlayers, decompositions);

#if 0
    const size_t nprocs = prod(nlayers, partitions_per_layer);
    for (size_t i = 0; i < nprocs; ++i) {
        size_t spatial_index[ndims * nlayers];
        to_spatial(i, ndims * nlayers, decompositions, spatial_index);
        size_t spatial_index_t[ndims * nlayers];
        transpose(spatial_index, nlayers, ndims, spatial_index_t);
        size_t decompositions_t[ndims * nlayers];
        transpose(decompositions, nlayers, ndims, decompositions_t);

        size_t gi = to_linear(ndims * nlayers, spatial_index_t, decompositions_t);
        size_t global_decomposition[ndims];
        contract(decompositions_t, ndims * nlayers, nlayers, global_decomposition);
        size_t global_index[ndims];
        to_spatial(gi, ndims, global_decomposition, global_index);
        // printf("Global index %zu\n", gi);
        // print("\tGlobal decomposition", ndims, global_decomposition);
        // print("\tGlobal index", ndims, global_index);

        // print("\tSpatial index", ndims * nlayers, spatial_index);
        // print("\tSpatial index transposed", ndims * nlayers, spatial_index_t);
        // print("\tDecomposition", ndims * nlayers, decompositions);
        // print("\tDecomposition transposed", ndims * nlayers, decompositions_t);

        // And backward
        gi = to_linear(ndims, global_index, global_decomposition);
        to_spatial(gi, ndims * nlayers, decompositions_t, spatial_index_t);
        transpose(spatial_index_t, ndims, nlayers, spatial_index);
        const size_t j = to_linear(ndims * nlayers, spatial_index, decompositions);

        // printf("\tNew Global index %zu\n", gi);
        // print("\tNew Spatial index transposed", ndims * nlayers, spatial_index_t);
        // print("\tNew Spatial index", ndims * nlayers, spatial_index);
        // printf("%zu vs. %zu\n", i, j);
        ERRCHK(i == j);
    }
#endif

    return 0;
}

AcDecompositionInfo
acDecompositionInfoCreate(const size_t ndims, const size_t* global_dims, //
                          const size_t nlayers, const size_t* partitions_per_layer)
{
    AcDecompositionInfo info = {
        .ndims                = ndims,
        .global_dims          = (size_t*)calloc(ndims, sizeof(info.global_dims[0])),
        .local_dims           = (size_t*)calloc(ndims, sizeof(info.local_dims[0])),
        .nlayers              = nlayers,
        .decomposition        = (size_t*)calloc(ndims * nlayers, sizeof(info.decomposition[0])),
        .global_decomposition = (size_t*)calloc(ndims, sizeof(info.global_decomposition[0])),
    };
    ERRCHK(info.global_dims);
    ERRCHK(info.local_dims);
    ERRCHK(info.decomposition);
    ERRCHK(info.global_decomposition);

    memcpy(info.global_dims, global_dims, ndims * sizeof(global_dims[0]));
    acHierarchicalDomainDecomposition(ndims, global_dims, nlayers, partitions_per_layer,
                                      info.local_dims, info.decomposition);

    size_t decomposition_transposed[ndims * nlayers];
    transpose(info.decomposition, nlayers, ndims, decomposition_transposed);
    contract(decomposition_transposed, ndims * nlayers, nlayers, info.global_decomposition);

    return info;
}

void
acDecompositionInfoDestroy(AcDecompositionInfo* info)
{
    free(info->global_dims);
    free(info->local_dims);
    free(info->decomposition);
    free(info->global_decomposition);
    memset(info, 0, sizeof(*info));
}

size_t
acGetPid(const size_t ndims, const int64_t pid_input[], const AcDecompositionInfo info)
{
    const size_t nlayers = info.nlayers;
    ERRCHK(ndims == info.ndims);

    int64_t global_decomposition[ndims];
    as_int64_t_array(ndims, info.global_decomposition, global_decomposition);

    int64_t pid_wrapped[ndims];
    mod_pointwise(ndims, pid_input, global_decomposition, pid_wrapped);

    size_t pid[ndims];
    as_size_t_array(ndims, pid_wrapped, pid);

    size_t gi = to_linear(ndims, pid, info.global_decomposition);

    size_t decomposition_transposed[ndims * nlayers];
    transpose(info.decomposition, nlayers, ndims, decomposition_transposed);

    size_t spatial_index_transposed[ndims * nlayers];
    size_t spatial_index[ndims * nlayers];
    to_spatial(gi, ndims * nlayers, decomposition_transposed, spatial_index_transposed);
    transpose(spatial_index_transposed, ndims, nlayers, spatial_index);

    const size_t j = to_linear(ndims * nlayers, spatial_index, info.decomposition);
    return j;
}

void
acGetPid3D(const size_t i, const AcDecompositionInfo info, const size_t ndims, int64_t pid_output[])
{
    const size_t nlayers = info.nlayers;
    ERRCHK(ndims == info.ndims);

    size_t spatial_index[ndims * nlayers];
    size_t spatial_index_transposed[ndims * nlayers];
    size_t decompositions_transposed[ndims * nlayers];
    size_t global_decomposition[ndims];
    size_t global_index[ndims];

    to_spatial(i, ndims * nlayers, info.decomposition, spatial_index);
    transpose(spatial_index, nlayers, ndims, spatial_index_transposed);
    transpose(info.decomposition, nlayers, ndims, decompositions_transposed);

    size_t gi = to_linear(ndims * nlayers, spatial_index_transposed, decompositions_transposed);
    contract(decompositions_transposed, ndims * nlayers, nlayers, global_decomposition);
    to_spatial(gi, ndims, global_decomposition, global_index);

    as_int64_t_array(ndims, global_index, pid_output);
}

// void
// acVerifyDecomposition(const AcDecompositionInfo info)
// {
//     ERRCHK(info.ndims == 3)
//     const size_t n = decomp.x * decomp.y * decomp.z; //
//     prod(info.ndims,info.global_decomposition); for (size_t i = 0; i < n; ++i)
//         ERRCHK(getPid(getPid3D(i, decomp), decomp) == i);

//     for (size_t k = 0; k < decomp.z; ++k) {
//         for (size_t j = 0; j < decomp.y; ++j) {
//             for (size_t i = 0; i < decomp.x; ++i) {

//                 const int3 center = {i, j, k};

//                 for (int k0 = -1; k0 <= 1; ++k0) {
//                     for (int j0 = -1; j0 <= 1; ++j0) {
//                         for (int i0 = -1; i0 <= 1; ++i0) {
//                             if (i0 == 0 && j0 == 0 && k0 == 0)
//                                 continue;
//                             int3 dir = (int3){i0, j0, k0};

//                             const int3 a = getPid3D(getPid(center + dir, decomp), decomp);
//                             const int3 b = getPid3D(getPid(a - dir, decomp), decomp);
//                             ERRCHK(b == center);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }