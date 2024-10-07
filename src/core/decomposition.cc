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
#include "decomposition.h"

#include <string.h> // memcpy

#define DECOMPOSITION_TYPE_ZORDER (1)
#define DECOMPOSITION_TYPE_HIERARCHICAL (2)
#if TWO_D == 0
#define MPI_DECOMPOSITION_AXES (3)
#else
#define MPI_DECOMPOSITION_AXES (2)
#endif

// #define DECOMPOSITION_TYPE (DECOMPOSITION_TYPE_HIERARCHICAL)

static void
acPrint_size_t(const char* label, const size_t value)
{
    printf("%s: %zu\n", label, value);
}

static void
acPrintArray_size_t(const char* label, const size_t count, const size_t* arr)
{
    printf("%s: (", label);
    for (size_t i = 0; i < count; ++i)
        printf("%zu%s", arr[i], i < count - 1 ? ", " : "");
    printf(")\n");
}

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

static size_t
prod(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

static void
factorize(const size_t n_initial, size_t* nfactors, size_t* factors)
{
    ERRCHK_ALWAYS(nfactors);
    size_t n     = n_initial;
    size_t count = 0;
    if (factors == NULL) {
        for (size_t i = 2; i <= n; ++i)
            while ((n % i) == 0) {
                ++count;
                n /= i;
            }
    }
    else {
        for (size_t i = 2; i <= n; ++i)
            while ((n % i) == 0) {
                factors[count++] = i;
                n /= i;
            }
    }
    *nfactors = count;
}

/** Requires that array is ordered */
static size_t
unique(const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count - 1; ++i)
        ERRCHK_ALWAYS(arr[i + 1] >= arr[i]);

    ERRCHK_ALWAYS(count > 0);
    size_t num_unique = 0;
    for (size_t i = 0; i < count; ++i) {
        arr[num_unique] = arr[i];
        ++num_unique;
        while ((i + 1 < count) && (arr[i + 1] == arr[i]))
            ++i;
    }

    return num_unique;
}

static void
transpose(const size_t* in, const size_t nrows, const size_t ncols, size_t* out)
{
    for (size_t i = 0; i < ncols; ++i) {
        for (size_t j = 0; j < nrows; ++j) {
            out[j + i * nrows] = in[i + j * ncols];
        }
    }
}

static void
contract(const size_t* in, const size_t length, const size_t factor, size_t* out)
{
    ERRCHK_ALWAYS((length % factor) == 0);
    const size_t out_length = length / factor;
    for (size_t j = 0; j < out_length; ++j) {
        out[j] = 1;
        for (size_t i = 0; i < factor; ++i)
            out[j] *= in[i + j * factor];
    }
}

static void
dims_create(const size_t nprocs, const size_t ndims, const size_t* dims, //
            size_t* local_dims, size_t* decomposition)
{
    // Do not allow aliasing
    ERRCHK_ALWAYS(dims < local_dims || dims > local_dims + ndims);

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
        ERRCHK_ALWAYS(bestj < ndims); // Failed to find proper dimensions
        ERRCHK_ALWAYS(prod(ndims, decomposition) <= nprocs);
        decomposition[bestj] *= bestfac;
        local_dims[bestj] /= bestfac;
    }
    free(factors);
    // printf("---COMPLETE--\n");

    ERRCHK_ALWAYS(prod(ndims, local_dims) * prod(ndims, decomposition) == prod(ndims, dims));
    fflush(stdout);
}

static void
to_spatial(const size_t index, const size_t ndims, const size_t* shape, size_t* output)
{
    for (size_t j = 0; j < ndims; ++j) {
        size_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        output[j] = (index / divisor) % shape[j];
    }
}

static size_t
to_linear(const size_t* index, const size_t ndims, const size_t* shape)
{
    size_t result = 0;
    for (size_t j = 0; j < ndims; ++j) {
        size_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += index[j] * factor;
    }
    return result;
}

static AcResult
acHierarchicalDomainDecomposition(const size_t ndims, const size_t* dims,                   //
                                  const size_t nlayers, const size_t* partitions_per_layer, //
                                  size_t* local_dims, size_t* decompositions)
{
    // print("Partitions per layer", nlayers, partitions_per_layer);
    // print("Dimensions", ndims, dims);

    size_t* global_dims = (size_t*) malloc(sizeof(size_t)*ndims);;
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
    free(global_dims);
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

        size_t gi = to_linear(spatial_index_t, ndims * nlayers, decompositions_t);
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
        gi = to_linear(global_index, ndims, global_decomposition);
        to_spatial(gi, ndims * nlayers, decompositions_t, spatial_index_t);
        transpose(spatial_index_t, ndims, nlayers, spatial_index);
        const size_t j = to_linear(spatial_index, ndims * nlayers, decompositions);

        // printf("\tNew Global index %zu\n", gi);
        // print("\tNew Spatial index transposed", ndims * nlayers, spatial_index_t);
        // print("\tNew Spatial index", ndims * nlayers, spatial_index);
        // printf("%zu vs. %zu\n", i, j);
        ERRCHK_ALWAYS(i == j);
    }
#endif

    return AC_SUCCESS;
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
    ERRCHK_ALWAYS(info.global_dims);
    ERRCHK_ALWAYS(info.local_dims);
    ERRCHK_ALWAYS(info.decomposition);
    ERRCHK_ALWAYS(info.global_decomposition);

    memcpy(info.global_dims, global_dims, ndims * sizeof(global_dims[0]));
    acHierarchicalDomainDecomposition(ndims, global_dims, nlayers, partitions_per_layer,
                                      info.local_dims, info.decomposition);

    size_t* decomposition_transposed = (size_t*)malloc(sizeof(size_t)*(ndims * nlayers));

    transpose(info.decomposition, nlayers, ndims, decomposition_transposed);
    contract(decomposition_transposed, ndims * nlayers, nlayers, info.global_decomposition);

    free(decomposition_transposed);

    acDecompositionInfoPrint(info);
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

static int64_t
mod(const int64_t a, const int64_t b)
{
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

static void
mod_pointwise(const size_t count, const int64_t* a, const int64_t* b, int64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = mod(a[i], b[i]);
}

static void
as_size_t_array(const size_t count, const int64_t* a, size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_size_t(a[i]);
}

static void
as_int64_t_array(const size_t count, const size_t* a, int64_t* b)
{
    for (size_t i = 0; i < count; ++i)
        b[i] = as_int64_t(a[i]);
}

int
acGetPid(const int3 pid_input, const AcDecompositionInfo info)
{
    const size_t ndims   = info.ndims;
    const size_t nlayers = info.nlayers;
    ERRCHK_ALWAYS(ndims == 3);

    const int64_t pid_unwrapped[] = {
        pid_input.x,
        pid_input.y,
        pid_input.z,
    };
    int64_t* global_decomposition = (int64_t*) malloc(sizeof(int64_t)*ndims);
    as_int64_t_array(ndims, info.global_decomposition, global_decomposition);

    int64_t* pid_wrapped = (int64_t*) malloc(sizeof(int64_t)*ndims);
    mod_pointwise(ndims, pid_unwrapped, global_decomposition, pid_wrapped);

    size_t* pid = (size_t*) malloc(sizeof(int64_t)*ndims);
    as_size_t_array(ndims, pid_wrapped, pid);

    size_t gi = to_linear(pid, ndims, info.global_decomposition);

    size_t* decomposition_transposed = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);
    transpose(info.decomposition, nlayers, ndims, decomposition_transposed);

    size_t* spatial_index_transposed = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);
    size_t* spatial_index            = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);
    to_spatial(gi, ndims * nlayers, decomposition_transposed, spatial_index_transposed);
    transpose(spatial_index_transposed, ndims, nlayers, spatial_index);

    const size_t j = to_linear(spatial_index, ndims * nlayers, info.decomposition);

    free(global_decomposition);
    free(pid_wrapped);
    free(decomposition_transposed);
    free(spatial_index_transposed);
    free(spatial_index);
    return as_int(j);
}

int3
acGetPid3D(const int i, const AcDecompositionInfo info)
{
    const size_t ndims   = info.ndims;
    const size_t nlayers = info.nlayers;
    ERRCHK_ALWAYS(ndims == 3);

    size_t* spatial_index            = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);
    size_t* spatial_index_transposed = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);
    size_t* decompositions_transposed = (size_t*) malloc(sizeof(int64_t)*ndims*nlayers);

    size_t* global_decomposition = (size_t*) malloc(sizeof(int64_t)*ndims);
    size_t* global_index         = (size_t*) malloc(sizeof(int64_t)*ndims);


    to_spatial(i, ndims * nlayers, info.decomposition, spatial_index);
    transpose(spatial_index, nlayers, ndims, spatial_index_transposed);
    transpose(info.decomposition, nlayers, ndims, decompositions_transposed);

    size_t gi = to_linear(spatial_index_transposed, ndims * nlayers, decompositions_transposed);
    contract(decompositions_transposed, ndims * nlayers, nlayers, global_decomposition);
    to_spatial(gi, ndims, global_decomposition, global_index);

    free(spatial_index);
    free(spatial_index_transposed);
    free(decompositions_transposed);

    free(global_decomposition);
    free(global_index);

    return (int3){
        as_int(global_index[0]),
        as_int(global_index[1]),
        as_int(global_index[2]),
    };
}


// --------------------
// Backwards compatibility
// --------------------
static AcDecompositionInfo g_decomposition_info;
static bool initialized = false;

void
compat_acDecompositionInit(const size_t ndims, const size_t* global_dims, //
                           const size_t nlayers, const size_t* partitions_per_layer)
{
    ERRCHK_ALWAYS(initialized == false);
    g_decomposition_info = acDecompositionInfoCreate(ndims, global_dims, nlayers,
                                                     partitions_per_layer);
    initialized          = true;
}
void
compat_acDecompositionQuit(void)
{
    if(initialized)
    	acDecompositionInfoDestroy(&g_decomposition_info);
    initialized = false;
}

static inline uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 3 * bit;
            k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
            i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
        }
    }
    // Just a quick copy/paste for other decomp dims
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 2 * bit;
#if TWO_D == 0
            j |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            k |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
#else
            i |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
#endif
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 1 * bit;
            k |= ((pid & (mask << 0)) >> 0 * bit) >> 0;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return (uint3_64){i, j, k};
}

static inline uint64_t
morton1D(const uint3_64 pid)
{
    uint64_t i = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 2 * bit;
            i |= ((pid.y & mask) << 1) << 2 * bit;
            i |= ((pid.x & mask) << 2) << 2 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
#if TWO_D == 0
            i |= ((pid.y & mask) << 0) << 1 * bit;
            i |= ((pid.z & mask) << 1) << 1 * bit;
#else
            i |= ((pid.x & mask) << 0) << 1 * bit;
            i |= ((pid.y & mask) << 1) << 1 * bit;
#endif
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 0 * bit;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return i;
}


static inline uint3_64
wrap(const int3 i, const uint3_64 n)
{
    return (uint3_64){
        mod(i.x, n.x),
        mod(i.y, n.y),
        mod(i.z, n.z),
    };
}

int
morton_getPid(const int3 pid_raw, const uint3_64 decomp)
{
    const uint3_64 pid = wrap(pid_raw, decomp);
    return (int)morton1D(pid);
}

int
linear_getPid(const int3 pid_raw, const uint3_64 decomp)
{
    const uint3_64 pid = wrap(pid_raw, decomp);
    return (int)pid.x + (int)pid.y*decomp.x + (int)pid.z*decomp.x*decomp.y;
}

int3
morton_getPid3D(const uint64_t pid, const uint3_64 decomp)
{
    const uint3_64 pid3D = morton3D(pid);
    ERRCHK_ALWAYS(morton_getPid(static_cast<int3>(pid3D), decomp) == (int)pid);
    return (int3){(int)pid3D.x, (int)pid3D.y, (int)pid3D.z};
}
int3
linear_getPid3D(const uint64_t pid, const uint3_64 decomp)
{
	return (int3){pid % decomp.x, (pid/decomp.x) % decomp.y, (pid/(decomp.x*decomp.y)) % decomp.z};
}

int
hierarchical_getPid(const int3 pid3D, const uint3_64 /* decomp */)
{
    ERRCHK_ALWAYS(initialized == true);
    return acGetPid(pid3D, g_decomposition_info);
}


int3
hierarchical_getPid3D(const uint64_t pid, const uint3_64 /* decomp*/)
{
    ERRCHK_ALWAYS(initialized == true);
    return acGetPid3D(pid, g_decomposition_info);
}


uint3_64
hierarchical_decompose(const uint64_t target)
{
    ERRCHK_ALWAYS(initialized == true);
    const uint3_64 p = (uint3_64){
        g_decomposition_info.global_decomposition[0],
        g_decomposition_info.global_decomposition[1],
        g_decomposition_info.global_decomposition[2]
    };
    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

uint3_64
morton_decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}


uint3_64
decompose(const uint64_t target, const AcDecomposeStrategy strategy)
{
	if(strategy == AcDecomposeStrategy::Morton)
		return morton_decompose(target);
	else if(strategy == AcDecomposeStrategy::Hierarchical)
		return hierarchical_decompose(target);
	return (uint3_64){0,0,0};
}

int
getPid(int3 pid, const uint3_64 decomp, const int proc_mapping_strategy)
{
	switch((AcProcMappingStrategy)proc_mapping_strategy)
	{
		case AcProcMappingStrategy::Linear:
			return linear_getPid(pid,decomp);
		case AcProcMappingStrategy::Morton:
			return morton_getPid(pid,decomp);
		case AcProcMappingStrategy::Hierarchical:
			return hierarchical_getPid(pid,decomp);
	}
	return -1;
}
int3
getPid3D(const uint64_t pid, const uint3_64 decomp, const int proc_mapping_strategy)
{
	switch((AcProcMappingStrategy)proc_mapping_strategy)
	{
		case AcProcMappingStrategy::Linear:
			return linear_getPid3D(pid,decomp);
		case AcProcMappingStrategy::Morton:
			return morton_getPid3D(pid,decomp);
		case AcProcMappingStrategy::Hierarchical:
			return hierarchical_getPid3D(pid,decomp);
	}
	return (int3){-1,-1,-1};
}

void
acVerifyDecomposition(const uint3_64 decomp, const int proc_mapping_strategy)
{
    const size_t n = decomp.x * decomp.y * decomp.z; // prod(info.ndims, info.global_decomposition);
    for (size_t i = 0; i < n; ++i)
        ERRCHK_ALWAYS(getPid(getPid3D(i, decomp,proc_mapping_strategy), decomp,proc_mapping_strategy) == (int)i);

    for (size_t k = 0; k < decomp.z; ++k) {
        for (size_t j = 0; j < decomp.y; ++j) {
            for (size_t i = 0; i < decomp.x; ++i) {

                const int3 center = {i, j, k};

                for (int k0 = -1; k0 <= 1; ++k0) {
                    for (int j0 = -1; j0 <= 1; ++j0) {
                        for (int i0 = -1; i0 <= 1; ++i0) {
                            if (i0 == 0 && j0 == 0 && k0 == 0)
                                continue;
                            int3 dir = (int3){i0, j0, k0};

                            const int3 a = getPid3D(getPid(center + dir, decomp,proc_mapping_strategy), decomp,proc_mapping_strategy);
                            const int3 b = getPid3D(getPid(a - dir, decomp,proc_mapping_strategy), decomp,proc_mapping_strategy);
                            ERRCHK_ALWAYS(b == center);
                        }
                    }
                }
            }
        }
    }
}
