#include "halo_segment_batch.h"

#include "dynamic_array.h"
#include "errchk.h"
#include "math_utils.h"
#include "matrix.h"
#include "ndarray.h"
#include "partition.h"
#include "print.h"

// static size_t
// get_halo_partitions(const size_t ndims, const size_t* mm, const size_t* nn, const size_t*
// nn_offset,
//                     const size_t npackets, size_t dims_matrix[npackets][ndims],
//                     size_t offset_matrix[npackets][ndims])
// {
//     if (npartitions == 0 || dims == NULL || offsets == NULL)
//         return partition(ndims, mm, nn, nn_offset, 0, NULL, NULL) - 1;

//     const size_t npartitions;
// }

HaloSegmentBatch
acHaloSegmentBatchCreate(const size_t ndims, const size_t* mm, const size_t* nn,
                         const size_t* nn_offset, const size_t nbuffers)
{
    // Determine the number of halo partitions
    const size_t npartitions = partition(ndims, mm, nn, nn_offset, 0, NULL, NULL);
    size_t dims_matrix[npartitions][ndims];
    size_t offset_matrix[npartitions][ndims];
    partition(ndims, mm, nn, nn_offset, npartitions, dims_matrix, offset_matrix);

    size_t npackets = 0;
    for (size_t i = 0; i < npartitions; ++i) {
        size_t dims[ndims];
        matrix_get_row(i, npartitions, ndims, dims_matrix, dims);
        size_t offset[ndims];
        matrix_get_row(i, npartitions, ndims, offset_matrix, offset);

        // Remove the packet with the computatinal domain
        // A more elegant way would probably be to check for an overlap with the
        // computational domain instead
        if (equals(ndims, nn, dims) && equals(ndims, nn_offset, offset)) {
            matrix_remove_row(i, npartitions, ndims, dims_matrix, dims_matrix);
            matrix_remove_row(i, npartitions, ndims, offset_matrix, offset_matrix);
            npackets = npartitions - 1;
            break;
        }
    }
    ERRCHK(npackets > 0);
    ERRCHK(npackets < npartitions);
    print_matrix("dims_matrix", npartitions, ndims, dims_matrix);
    print_matrix("offset_matrix", npartitions, ndims, offset_matrix);

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
        const size_t* dims      = dims_matrix[i];
        const size_t* offset    = offset_matrix[i];
        batch.local_packets[i]  = acHaloSegmentCreate(ndims, dims, offset, nbuffers);
        batch.remote_packets[i] = acHaloSegmentCreate(ndims, dims, offset, nbuffers);
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
        acHaloSegmentDestroy(&batch->local_packets[i]);
        acHaloSegmentDestroy(&batch->remote_packets[i]);
    }
    free(batch->remote_packets);
    free(batch->local_packets);
    batch->npackets = 0;
}
