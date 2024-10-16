#include "segment_batch.h"

static void halo_segment_batch_verify(const size_t ndims, const uint64_t* mm, const uint64_t* nn,
                                      const struct HaloSegmentBatch_s batch);

/** Returns the MPI rank of the neighbor that is responsible for the data at offset.
 * Visual example:
 *        |---|--------|---|
 *        0   rr       nn  mm
 * rank:   i-1     i    i+1
 *
 */
static void
get_mpi_send_recv_peers(const size_t ndims, const int* nn, const int* rr, const int* offset,
                        int* send_peer, int* recv_peer)
{
    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));

    int* coords = ac_calloc(ndims, sizeof(coords[0]));
    ERRCHK_MPI_API(MPI_Cart_coords(mpi_comm_, rank, as_int(ndims), coords));

    int *recv_coords, *send_coords;
    ndup(ndims, coords, recv_coords);
    ndup(ndims, coords, send_coords);

    for (size_t i = 0; i < ndims; ++i) {
        recv_coords[i] += offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
        send_coords[i] -= offset[i] < rr[i] ? -1 : offset[i] >= rr[i] + nn[i] ? 1 : 0;
    }

    ERRCHK_MPI_API(MPI_Cart_rank(mpi_comm_, recv_coords, recv_peer));
    ERRCHK_MPI_API(MPI_Cart_rank(mpi_comm_, send_coords, send_peer));

    ndealloc(send_coords);
    ndealloc(recv_coords);
    ndealloc(coords);
}

struct HaloSegmentBatch_s*
halo_segment_batch_create(const size_t ndims, const uint64_t* local_mm, const uint64_t* local_nn,
                          const uint64_t* local_nn_offset, const uint64_t n_aggregate_buffers)
{
    // Partition the domain
    SegmentArray segments;
    dynarr_create_with_destructor(segment_destroy, &segments);
    partition(ndims, local_mm, local_nn, local_nn_offset, &segments);

    // Prune partitions within the computational domain
    for (size_t i = 0; i < segments.length; ++i) {
        if (within_box_note_changed(ndims, segments.data[i].offset, local_nn, local_nn_offset)) {
            dynarr_remove(i, &segments);
            --i;
        }
    }

    struct HaloSegmentBatch_s* batch;
    nalloc(1, batch);

    batch->ndims = ndims;
    ndup(ndims, local_mm, batch->local_mm);
    ndup(ndims, local_nn, batch->local_nn);
    ndup(ndims, local_nn_offset, batch->local_nn_offset);

    batch->npackets = segments.length;
    nalloc(batch->npackets, batch->local_packets);
    nalloc(batch->npackets, batch->remote_packets);

    for (size_t i = 0; i < segments.length; ++i) {
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
        //
        //
        // Send phase
        // process 1   |---|-----ooo|---|
        // process 2            |---|-----ooo|---|
        // process 3                     |---|-----ooo|---|
        //
        // Recv phase
        // process 1   |ooo|--------|---|
        // process 2            |ooo|--------|---|
        // process 3                     |ooo|--------|---|
        //
        //
        // Send always forward
        // Recv always from behind
        //
        // recv offset tells the direction where the message comes from
        // its inverse direction is the sending direction
        // send direction cannot be determined from the send offset, because
        // send offsets are not unique.
        const uint64_t* subdims = segments.data[i].dims;

        // Recv packet
        const uint64_t* recv_offset = segments.data[i].offset;
        batch->remote_packets[i] = packet_create(ndims, subdims, recv_offset, n_aggregate_buffers);

        // Send packet
        uint64_t* send_offset = ac_calloc(ndims, sizeof(send_offset[0]));
        for (size_t j = 0; j < ndims; ++j)
            send_offset[j] = ((local_nn[j] + recv_offset[j] - local_nn_offset[j]) % local_nn[j]) +
                             local_nn_offset[j];
        batch->local_packets[i] = packet_create(ndims, subdims, send_offset, n_aggregate_buffers);

        ndealloc(send_offset);
    }

    // Cleanup
    dynarr_destroy(&segments);

    // Sanity check: verify that the segments are within the expected bounds
    halo_segment_batch_verify(ndims, local_mm, local_nn, *batch);
    return batch;
}

void
halo_segment_batch_destroy(struct HaloSegmentBatch_s** batch)
{
    ndealloc((*batch)->local_nn_offset);
    ndealloc((*batch)->local_nn);
    ndealloc((*batch)->local_mm);
    (*batch)->ndims = 0;
    for (size_t i = 0; i < (*batch)->npackets; ++i) {
        packet_destroy(&(*batch)->local_packets[i]);
        packet_destroy(&(*batch)->remote_packets[i]);
    }
    ndealloc((*batch)->local_packets);
    ndealloc((*batch)->remote_packets);
    (*batch)->npackets = 0;
    ndealloc((*batch));
}

// static void
// halo_segment_batch_pack(const uint64_t ninputs, double** inputs, HaloSegmentBatch* batch)
// {
//     const size_t ndims     = batch->ndims;
//     const uint64_t* local_mm = batch->local_mm;

//     uint64_t* zeros= ac_calloc(ndims, sizeof(zeros[0]));

//     for (size_t i = 0; i < batch->npackets; ++i) {
//         for (size_t j = 0; j < ninputs; ++j) {
//             Packet* packet              = &batch->local_packets[i];
//             const uint64_t* input_dims    = local_mm;
//             const uint64_t* input_offset  = packet->segment.offset;
//             const double* input         = inputs[j];
//             const uint64_t* output_dims   = packet->segment.dims;
//             const uint64_t* output_offset = zeros;
//             double* output              = &packet->buffer
//                                   .data[j * prod(packet->segment.ndims, packet->segment.dims)];
//             pack(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
//         }
//     }
//     ndealloc(zeros);
// }

// static void
// halo_segment_batch_unpack(const HaloSegmentBatch* batch, const uint64_t noutputs, double**
// outputs)
// {
//     const size_t ndims     = batch->ndims;
//     const uint64_t* local_mm = batch->local_mm;

//     uint64_t* zeros= ac_calloc(ndims, sizeof(zeros[0]));

//     for (size_t i = 0; i < batch->npackets; ++i) {
//         for (size_t j = 0; j < noutputs; ++j) {
//             Packet* packet              = &batch->remote_packets[i];
//             const uint64_t* output_dims   = local_mm;
//             const uint64_t* output_offset = packet->segment.offset;
//             double* output              = outputs[j];
//             const uint64_t* input_dims    = packet->segment.dims;
//             const uint64_t* input_offset  = zeros;
//             const double* input         = &packet->buffer
//                                        .data[j * prod(packet->segment.ndims,
//                                        packet->segment.dims)];
//             pack(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
//         }
//     }
//     ndealloc(zeros);
// }

void
halo_segment_batch_launch(const uint64_t ninputs, double* inputs[],
                          struct HaloSegmentBatch_s* batch)
{
    const size_t ndims              = batch->ndims;
    const uint64_t* local_mm        = batch->local_mm;
    const uint64_t* local_nn        = batch->local_nn;
    const uint64_t* local_nn_offset = batch->local_nn_offset;
    for (size_t i = 0; i < batch->npackets; ++i) {

        // Packets
        Packet* local_packet  = &batch->local_packets[i];
        Packet* remote_packet = &batch->remote_packets[i];
        const int tag         = get_tag();

        int rank;
        ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));

        // Pack
        pack(batch->ndims, batch->local_mm, local_packet->segment.dims,
             local_packet->segment.offset, ninputs, inputs, local_packet->buffer.data);

        // Error check
        ERRCHK_MPI(as_int(remote_packet->buffer.count) == as_int(local_packet->buffer.count));
        const size_t count = ninputs *
                             prod(local_packet->segment.ndims, local_packet->segment.dims);
        ERRCHK_MPI(count == local_packet->buffer.count);

        int send_peer, recv_peer;

        int* mpi_nn     = as_mpi_format_alloc(ndims, local_nn);
        int* mpi_rr     = as_mpi_format_alloc(ndims, local_nn_offset);
        int* mpi_offset = as_mpi_format_alloc(ndims, remote_packet->segment.offset);
        get_mpi_send_recv_peers(ndims, mpi_nn, mpi_rr, mpi_offset, &send_peer, &recv_peer);
        ndealloc(mpi_nn);
        ndealloc(mpi_rr);
        ndealloc(mpi_offset);

        // Post recv
        ERRCHK_MPI_API(MPI_Irecv(remote_packet->buffer.data, as_int(remote_packet->buffer.count),
                                 mpi_dtype_, recv_peer, tag, mpi_comm_, &remote_packet->req));

        // Post send
        ERRCHK_MPI_API(MPI_Isend(local_packet->buffer.data, as_int(local_packet->buffer.count),
                                 mpi_dtype_, send_peer, tag, mpi_comm_, &local_packet->req));

        // MPI_Status status;
        // ERRCHK_MPI_API(MPI_Sendrecv(local_packet->buffer.data,
        // as_int(local_packet->buffer.count),
        //                             mpi_dtype_, local_packet->peer, tag,
        //                             remote_packet->buffer.data,
        //                             as_int(remote_packet->buffer.count), mpi_dtype_,
        //                             remote_packet->peer, tag, mpi_comm_, &status));
    }
}

void
halo_segment_batch_wait(struct HaloSegmentBatch_s* batch, const uint64_t noutputs,
                        double* outputs[])
{
    for (size_t i = 0; i < batch->npackets; ++i) {
        packet_wait(&batch->local_packets[i]);
        packet_wait(&batch->remote_packets[i]);

        Packet* remote_packet = &batch->remote_packets[i];
        unpack(remote_packet->buffer.data, batch->ndims, batch->local_mm,
               remote_packet->segment.dims, remote_packet->segment.offset, noutputs, outputs);
    }
}

static void
halo_segment_batch_verify(const size_t ndims, const uint64_t* mm, const uint64_t* nn,
                          const struct HaloSegmentBatch_s batch)
{
    // Ensure that
    // 1. none of the halo segments (remote_packets) overlap
    // 2. the number of points covered by the segments encompass the whole
    //    ghost zone
    // 3. all of the segments are within mm
    // 4. all of the outer computational domain (local_packets) are within nn

    ERRCHK_MPI(batch.ndims == ndims);
    { // Local packets
        const uint64_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const Segment a = batch.local_packets[i].segment;
            count += prod(ndims, a.dims);

            ERRCHK_MPI(within_box_note_changed(ndims, a.offset, a.dims, a.offset));
            for (size_t j = 0; j < ndims; ++j)
                ERRCHK_MPI(a.dims[j] <= nn[j]);
        }
        ERRCHK_MPI(count == model_count);
    }
    { // Remote packets
        const uint64_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const Segment a = batch.remote_packets[i].segment;
            ERRCHK_MPI(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const Segment b = batch.remote_packets[j].segment;

                ERRCHK_MPI(intersect_box_note_changed(ndims, a.dims, a.offset, b.dims, b.offset) ==
                           false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK_MPI(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK_MPI(count == model_count);
    }
}

static void
test_halo_segment_batch(void)
{
    const uint64_t mm[]        = {8, 8};
    const uint64_t nn[]        = {6, 6, 6};
    const uint64_t nn_offset[] = {1, 1, 1};
    const size_t ndims         = ARRAY_SIZE(mm);
    const uint64_t ninputs     = 3;

    HaloSegmentBatch batch = halo_segment_batch_create(ndims, mm, nn, nn_offset, ninputs);

    double** inputs = ac_calloc(ninputs, sizeof(inputs[0]));
    for (size_t i = 0; i < ninputs; ++i)
        nalloc(prod(ndims, mm), inputs[i]);

    // halo_segment_batch_pack(ninputs, inputs, &batch);
    // halo_segment_batch_unpack(&batch, ninputs, inputs);

    for (size_t i = 0; i < ninputs; ++i)
        ndealloc(inputs[i]);
    ndealloc(inputs);

    // halo_segment_batch_wait(&batch);
    // printf("Created halo segments\n");
    // print("batch.local_packets.length", batch.local_packets.length);
    // for (size_t i = 0; i < batch.local_packets.length; ++i) {
    //     printf("%zu:\n", i);
    //     print_packet("Send", batch.local_packets.data[i]);
    //     print_packet("Recv", batch.remote_packets.data[i]);
    // }

    halo_segment_batch_destroy(&batch);
}
