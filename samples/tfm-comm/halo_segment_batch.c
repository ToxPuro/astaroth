#include "halo_segment_batch.h"

#include "dynamic_array.h"
#include "errchk_mpi.h"
#include "math_utils.h"
#include "matrix.h"
#include "mpi_utils.h"
#include "ndarray.h"
#include "partition.h"
#include "print.h"
#include "type_conversion.h"

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
            ERRCHK_MPI(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.local_packets[j];

                ERRCHK_MPI(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK_MPI(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK_MPI(count == model_count);
    }
    { // Remote packets
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const HaloSegment a = batch.remote_packets[i];
            ERRCHK_MPI(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.remote_packets[j];

                ERRCHK_MPI(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK_MPI(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK_MPI(count == model_count);
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
    ERRCHK_MPI(equals(ndims, nn, dims_matrix[npackets]) &&
               equals(ndims, nn_offset, offset_matrix[npackets]));
    print_matrix("dims_matrix", npackets, ndims, dims_matrix);
    print_matrix("offset_matrix", npackets, ndims, offset_matrix);

    // Create HaloSegmentBatch
    HaloSegmentBatch batch = (HaloSegmentBatch){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(batch.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(batch.remote_packets[0]) * npackets),
        .send_reqs      = malloc(sizeof(batch.send_reqs[0]) * npackets),
        .recv_reqs      = malloc(sizeof(batch.recv_reqs[0]) * npackets),
        .send_subarrays = malloc(sizeof(batch.send_subarrays[0]) * npackets),
        .recv_subarrays = malloc(sizeof(batch.recv_subarrays[0]) * npackets),
    };
    ERRCHK_MPI(batch.local_packets);
    ERRCHK_MPI(batch.remote_packets);
    ERRCHK_MPI(batch.send_reqs);
    ERRCHK_MPI(batch.recv_reqs);
    ERRCHK_MPI(batch.send_subarrays);
    ERRCHK_MPI(batch.recv_subarrays);

    MPI_Datatype dtype = MPI_UNSIGNED_LONG_LONG;
    WARNING("MPI_Datatype hardcoded to MPI_UNSIGNED_LONG_LONG");
    for (size_t i = 0; i < npackets; ++i) {
        const size_t* dims        = dims_matrix[i];
        const size_t* recv_offset = offset_matrix[i];
        batch.local_packets[i]    = acHaloSegmentCreate(ndims, dims, recv_offset, nbuffers);
        batch.remote_packets[i]   = acHaloSegmentCreate(ndims, dims, recv_offset, nbuffers);

        // Subarrays
        int sizes[ndims], subsizes[ndims], recv_starts[ndims], send_starts[ndims];
        to_mpi_format(ndims, mm, sizes);
        to_mpi_format(ndims, dims, subsizes);
        to_mpi_format(ndims, recv_offset, recv_starts);

        // Each recv_offset maps to a coordinates in the computational domain of a neighboring
        // process, written as send_offset.
        //
        // The equation is
        //      a = (b - r) mod n + r,
        //  where a is the send_offset, b the recv_offset, r the nn_offset, and n the size of the
        //  computational domain.
        //
        // Intuitively:
        //  1. Map b to coordinates in the halo-less computational domain
        //  2. Wrap the resulting out-of-bounds coordinates around the computational domain
        //  3. Map the result back to coordinates in the buffer that includes the halo
        //
        // The computation is implemented as a = (n + b - r) mod n + r to avoid
        // unsigned integer underflow.
        size_t send_offset[ndims];
        for (size_t j = 0; j < ndims; ++j)
            send_offset[j] = ((nn[j] + recv_offset[j] - nn_offset[j]) % nn[j]) + nn_offset[j];
        to_mpi_format(ndims, send_offset, send_starts);

        // Subarrays: receive
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, recv_starts,
                                                MPI_ORDER_C, dtype, &batch.recv_subarrays[i]));
        ERRCHK_MPI_API(MPI_Type_commit(&batch.recv_subarrays[i]));

        // Subarrays: send
        ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), sizes, subsizes, send_starts,
                                                MPI_ORDER_C, dtype, &batch.send_subarrays[i]));
        ERRCHK_MPI_API(MPI_Type_commit(&batch.send_subarrays[i]));
    }

    acHaloSegmentBatchVerify(ndims, mm, nn, batch);
    return batch;
}

void
acHaloSegmentBatchDestroy(HaloSegmentBatch* batch)
{

    for (size_t i = 0; i < batch->npackets; ++i) {
        ERRCHK_MPI_API(MPI_Type_free(&batch->send_subarrays[i]));
        ERRCHK_MPI_API(MPI_Type_free(&batch->recv_subarrays[i]));
        acHaloSegmentDestroy(&batch->local_packets[i]);
        acHaloSegmentDestroy(&batch->remote_packets[i]);
    }
    free(batch->recv_subarrays);
    free(batch->send_subarrays);
    free(batch->send_reqs);
    free(batch->recv_reqs);
    free(batch->remote_packets);
    free(batch->local_packets);
    batch->npackets = 0;
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
acHaloSegmentBatchWait(const HaloSegmentBatch batch)
{
    MPI_Status send_statuses[batch.npackets];
    ERRCHK_MPI_API(MPI_Waitall(as_int(batch.npackets), batch.send_reqs, send_statuses));
    for (size_t i = 0; i < batch.npackets; ++i)
        ERRCHK_MPI_API(send_statuses[i].MPI_ERROR);

    MPI_Status recv_statuses[batch.npackets];
    ERRCHK_MPI_API(MPI_Waitall(as_int(batch.npackets), batch.recv_reqs, recv_statuses));
    for (size_t i = 0; i < batch.npackets; ++i)
        ERRCHK_MPI_API(recv_statuses[i].MPI_ERROR);
}
