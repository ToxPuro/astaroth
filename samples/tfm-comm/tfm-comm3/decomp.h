#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t ndims;
    uint64_t* global_nn; // [ndims]
    uint64_t* local_nn;  // [ndims]

    size_t nlayers;
    uint64_t* decomposition;        // [nlayers * ndims]
    uint64_t* global_decomposition; // [ndims]
} AcDecompositionInfo;

AcDecompositionInfo acDecompositionInfoCreate(const size_t ndims, const uint64_t* global_nn,
                                              const size_t nlayers,
                                              const uint64_t* partitions_per_layer);

void acDecompositionInfoDestroy(AcDecompositionInfo* info);

void acDecompositionInfoPrint(const AcDecompositionInfo info);

uint64_t acDecompositionGetRank(const size_t ndims, const uint64_t* coords,
                                const AcDecompositionInfo info);

void acDecompositionGetCoords(const uint64_t rank, const AcDecompositionInfo info, size_t ndims,
                              uint64_t* coords);
