#include "decomp.h"

#include "alloc.h"
#include "errchk.h"
#include "math_utils.h"
#include "print.h"

static void
dims_create(const size_t nprocs, const size_t ndims, const uint64_t* global_nn, uint64_t* local_nn,
            uint64_t* decomposition)
{
    ac_copy(ndims, sizeof(global_nn[0]), global_nn, local_nn);
    array_set_uint64_t(1, ndims, decomposition);

    if (nprocs == 1) return;

    size_t nfactors;
    factorize(nprocs, &nfactors, NULL);
    uint64_t* factors = ac_calloc(nfactors, sizeof(factors[0]));
    factorize(nprocs, &nfactors, factors);

    while (prod(ndims, decomposition) != nprocs) {
        // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
        // Adapted to try out all factors to work with a wider range of dims
        // Or maybe this is what they meant all along, but the description was just unclear
        size_t bestj          = SIZE_MAX;
        uint64_t bestlocal_nn = 0;
        uint64_t bestfac      = 0;
        for (size_t i = 0; i < nfactors; ++i) {
            const uint64_t fac = factors[nfactors - 1 - i];
            // printf("fac %zu\n", fac);
            // print("local dims", ndims, local_nn);
            for (size_t j = 0; j < ndims; ++j) {
                // if ((local_nn[j] % fac) == 0)
                //     printf("\t[%zu] = %zu divisible by %zu\n", j, local_nn[j], fac);
                if ((local_nn[j] % fac) == 0 && (local_nn[j] > bestlocal_nn)) {
                    bestj        = j;
                    bestlocal_nn = local_nn[j];
                    bestfac      = fac;
                }
            }
        }
        // printf("chose [%zu] = %zu\n", bestj, bestlocal_nn);
        ERRCHK(bestj < ndims); // Failed to find proper dimensions
        ERRCHK(prod(ndims, decomposition) <= nprocs);
        decomposition[bestj] *= bestfac;
        local_nn[bestj] /= bestfac;
    }
    ac_free(factors);

    ERRCHK(prod(ndims, local_nn) * prod(ndims, decomposition) == prod(ndims, global_nn));
}

static void
hierarchical_domain_decomposition(const size_t ndims, const uint64_t* global_nn, //
                                  const size_t nlayers,
                                  const uint64_t* partitions_per_layer, //
                                  uint64_t* local_nn, uint64_t* decompositions)
{
    uint64_t* curr_global_nn = ac_dup(ndims, sizeof(curr_global_nn[0]), global_nn);

    for (size_t i = nlayers - 1; i < nlayers; --i) {
        dims_create(partitions_per_layer[i], ndims, curr_global_nn, local_nn,
                    &decompositions[i * ndims]);
        ac_copy(ndims, sizeof(local_nn[0]), local_nn, curr_global_nn);
    }

    ac_free(curr_global_nn);
}

AcDecompositionInfo
acDecompositionInfoCreate(const size_t ndims, const uint64_t* global_nn, const size_t nlayers,
                          const uint64_t* partitions_per_layer)
{
    AcDecompositionInfo info = {
        .ndims                = ndims,
        .global_nn            = ac_dup(ndims, sizeof(info.global_nn[0]), global_nn),
        .local_nn             = ac_calloc(ndims, sizeof(info.local_nn[0])),
        .nlayers              = nlayers,
        .decomposition        = ac_calloc(ndims * nlayers, sizeof(info.decomposition[0])),
        .global_decomposition = ac_calloc(ndims, sizeof(info.global_decomposition[0])),
    };

    hierarchical_domain_decomposition(ndims, global_nn, nlayers, partitions_per_layer,
                                      info.local_nn, info.decomposition);

    acDecompositionInfoPrint(info);
    return info;
}

void
acDecompositionInfoDestroy(AcDecompositionInfo* info)
{
    ac_free(info->global_decomposition);
    ac_free(info->decomposition);
    ac_free(info->local_nn);
    ac_free(info->global_nn);
    info->nlayers = 0;
    info->ndims   = 0;
}

void
acDecompositionInfoPrint(const AcDecompositionInfo info)
{
    printf("AcDecompositionInfo %p\n", &info);
    print("\tndims", info.ndims);
    print("\tnlayers", info.nlayers);
    print_array("\tglobal_nn", info.ndims, info.global_nn);
    print_array("\tlocal_nn", info.ndims, info.local_nn);
    print_array("\tdecomposition", info.ndims * info.nlayers, info.decomposition);
    print_array("\tglobal_decomposition", info.ndims, info.global_decomposition);
}

uint64_t
acDecompositionGetRank(const size_t ndims, const uint64_t* coords, const AcDecompositionInfo info)
{
    return 0;
    // ERRCHK(ndims == info.ndims);
    // const uint64_t nlayers = info.nlayers;

    // uint64_t* nd_coords = ac_calloc(nlayers * ndims, sizeof(nd_coords[0]));
    // for (size_t i = 0; i < ndims; ++i) {
    //     to_spatial()
    // }
    // ac_free(nd_coords);

    //     const uint64_t nlayers = info.nlayers;

    //     // Get linear index in the lower domain
    //    const  uint64_t gi = to_linear(ndims, coords, info.global_decomposition);

    //     // Project it to N dimensions
    //     uint64_t* nd_coords = ac_calloc(ndims * nlayers, sizeof(nd_coords[0]));
    //     to_spatial(gi, ndims * nlayers, info.decomposition, nd_coords);
    //     const uint64_t i = to_linear()

    //     ac_free(nd_coords);

    //     return to_linear(ndims, coords, info.decomposition);
}

void
acDecompositionGetCoords(const uint64_t rank, const AcDecompositionInfo info, size_t ndims,
                         uint64_t* coords)
{
}
