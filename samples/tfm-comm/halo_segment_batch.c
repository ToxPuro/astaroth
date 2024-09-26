#include "halo_segment_batch.h"

#include "dynamic_array.h"
#include "errchk.h"
#include "math_utils.h"
#include "matrix.h"
#include "ndarray.h"
#include "partition.h"
#include "print.h"

static void
acHaloSegmentBatchVerify(const size_t ndims, const size_t* mm, const size_t* nn,
                         const HaloSegmentBatch batch)
{
    // Ensure that
    // 1. none of the segments overlap
    // 2. the number of points covered by the segments encompass the whole
    //    ghost zone
    // 3. all of the segments are within mm

    { // Local packets
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const HaloSegment a = batch.local_packets[i];
            ERRCHK(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.local_packets[j];

                ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK(count == model_count);
    }
    { // Remote packets
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const HaloSegment a = batch.remote_packets[i];
            ERRCHK(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.remote_packets[j];

                ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK(count == model_count);
    }
}

HaloSegmentBatch
acHaloSegmentBatchCreate(const size_t ndims, const size_t* mm, const size_t* nn,
                         const size_t* nn_offset, const size_t nbuffers)
{
    // Determine the number of halo partitions
    const size_t npartitions = partition(ndims, mm, nn, nn_offset, 0, NULL, NULL);
    size_t dims_matrix[npartitions][ndims], offset_matrix[npartitions][ndims];
    partition(ndims, mm, nn, nn_offset, npartitions, dims_matrix, offset_matrix);

    // Quick solution: assume the last partition is the innermost domain
    // and that the other partitions do not overlap the computational domain
    const size_t npackets = npartitions - 1;
    ERRCHK(equals(ndims, nn, dims_matrix[npackets]) &&
           equals(ndims, nn_offset, offset_matrix[npackets]));
    print_matrix("dims_matrix", npackets, ndims, dims_matrix);
    print_matrix("offset_matrix", npackets, ndims, offset_matrix);

    // Create HaloSegmentBatch
    HaloSegmentBatch batch = (HaloSegmentBatch){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(batch.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(batch.remote_packets[0]) * npackets),
    };
    ERRCHK(batch.local_packets);
    ERRCHK(batch.remote_packets);

    for (size_t i = 0; i < npackets; ++i) {
        const size_t* dims      = dims_matrix[i];
        const size_t* offset    = offset_matrix[i];
        batch.local_packets[i]  = acHaloSegmentCreate(ndims, dims, offset, nbuffers);
        batch.remote_packets[i] = acHaloSegmentCreate(ndims, dims, offset, nbuffers);
    }

    acHaloSegmentBatchVerify(ndims, mm, nn, batch);
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
        acHaloSegmentDestroy(&batch->local_packets[i]);
        acHaloSegmentDestroy(&batch->remote_packets[i]);
    }
    free(batch->remote_packets);
    free(batch->local_packets);
    batch->npackets = 0;
}
