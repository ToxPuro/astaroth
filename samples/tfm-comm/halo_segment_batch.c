#include "halo_segment_batch.h"

#include "dynamic_array.h"
#include "errchk.h"
#include "math_utils.h"
#include "matrix.h"
#include "ndarray.h"
#include "partition.h"
#include "print.h"

HaloSegmentBatch
acHaloSegmentBatchCreate(const size_t ndims, const size_t* mm, const size_t* nn,
                         const size_t* nn_offset, const size_t nfields)
{
    // Determine the number of halo partitions
    const size_t npartitions = partition(ndims, mm, nn, nn_offset, 0, NULL, NULL);
    size_t dims[npartitions][ndims];
    size_t offsets[npartitions][ndims];
    partition(ndims, mm, nn, nn_offset, npartitions, dims, offsets);
    print_matrix("Dims in segment", npartitions, ndims, dims);
    print_matrix("offsets in segment", npartitions, ndims, offsets);
    // TODO REMOVE THE computational domain MIDDLE and determine npackets
    size_t npackets = 0;
    WARNING("TODO determine how to remove center block")

    print("Creating npackets", npackets);

    // Create HaloSegmentBatch
    HaloSegmentBatch batch = (HaloSegmentBatch){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(batch.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(batch.remote_packets[0]) * npackets),
    };
    ERRCHK(batch.local_packets);
    ERRCHK(batch.remote_packets);

    for (size_t i = 0; i < npackets; ++i) {
        const size_t dims[]     = {3, 3, 3};
        const size_t offset[]   = {0, 0, 0};
        batch.local_packets[i]  = acCreateHaloSegment(ndims, dims, offset, nfields);
        batch.remote_packets[i] = acCreateHaloSegment(ndims, dims, offset, nfields);
    }

    return batch;
}

void
acHaloSegmentBatchPrint(const char* label, const HaloSegmentBatch batch)
{
    printf("HaloSegmentBatch %s:\n", label);
    print("\tnpackets", batch.npackets);

    const size_t buflen = 128;
    char buf[buflen];

    for (size_t i = 0; i < batch.npackets; ++i) {
        snprintf(buf, buflen, "local_packets[%zu]", i);
        acHaloSegmentPrint(buf, batch.local_packets[i]);
    }

    for (size_t i = 0; i < batch.npackets; ++i) {
        snprintf(buf, buflen, "remote_packets[%zu]", i);
        acHaloSegmentPrint(buf, batch.remote_packets[i]);
    }
}

void
acHaloSegmentBatchDestroy(HaloSegmentBatch* batch)
{
    for (size_t i = 0; i < batch->npackets; ++i) {
        acDestroyHaloSegment(&batch->local_packets[i]);
        acDestroyHaloSegment(&batch->remote_packets[i]);
    }
    free(batch->remote_packets);
    free(batch->local_packets);
    batch->npackets = 0;
}
