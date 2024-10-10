#include "comm.h"

#include <mpi.h>

#include "buf.h"
#include "errchk_mpi.h"
#include "math_utils.h"
#include "misc.h"
#include "mpi_utils.h"
#include "nalloc.h"
#include "print.h"
#include "segment.h"
#include "type_conversion.h"

#define MPI_SYNCHRONOUS_BLOCK_START                                                                \
    {                                                                                              \
        fflush(stdout);                                                                            \
        MPI_Barrier(mpi_comm_);                                                                    \
        int rank__, nprocs_;                                                                       \
        ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank__));                                         \
        ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, &nprocs_));                                        \
        for (int i__ = 0; i__ < nprocs_; ++i__) {                                                  \
            if (i__ == rank__) {                                                                   \
                printf("---Rank %d---\n", rank__);

#define MPI_SYNCHRONOUS_BLOCK_END                                                                  \
    }                                                                                              \
    fflush(stdout);                                                                                \
    MPI_Barrier(mpi_comm_);                                                                        \
    }                                                                                              \
    MPI_Barrier(mpi_comm_);                                                                        \
    }

/*
 * Comm
 */
static MPI_Comm mpi_comm_            = MPI_COMM_NULL;
static int mpi_ndims_                = -1;
static const MPI_Datatype mpi_dtype_ = MPI_DOUBLE;

void
acCommInit(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
}

void
acCommSetup(const size_t ndims, const size_t* global_nn, size_t* local_nn, size_t* global_nn_offset)
{
    // Get nprocs
    int nprocs;
    ERRCHK_MPI_API(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // Decompose
    int* mpi_dims;
    ncalloc(ndims, mpi_dims);
    ERRCHK_MPI_API(MPI_Dims_create(nprocs, as_int(ndims), mpi_dims));

    // Create the primary communicator
    int* mpi_periods;
    nalloc(ndims, mpi_periods);
    iset(1, ndims, mpi_periods);
    ERRCHK_MPI_API(
        MPI_Cart_create(MPI_COMM_WORLD, as_int(ndims), mpi_dims, mpi_periods, 0, &mpi_comm_));

    // Set errors as non-fatal
    ERRCHK_MPI_API(MPI_Comm_set_errhandler(mpi_comm_, MPI_ERRORS_RETURN));

    // Setup the rest of the global variables
    mpi_ndims_ = as_int(ndims);

    // Compute local_nn
    size_t* dims;
    nalloc(ndims, dims);
    to_astaroth_format(ndims, mpi_dims, dims);
    for (size_t i = 0; i < ndims; ++i)
        local_nn[i] = global_nn[i] / dims[i];
    ndealloc(dims);

    // Compute global_nn_offset
    int rank;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));

    int* mpi_coords;
    ncalloc(ndims, mpi_coords);
    ERRCHK_MPI_API(MPI_Cart_coords(mpi_comm_, rank, mpi_ndims_, mpi_coords));

    size_t* coords;
    nalloc(ndims, coords);
    to_astaroth_format(ndims, mpi_coords, coords);
    for (size_t i = 0; i < ndims; ++i)
        global_nn_offset[i] = local_nn[i] * coords[i];

    // Cleanup
    ndealloc(coords);
    ndealloc(mpi_coords);
    ndealloc(mpi_periods);
    ndealloc(mpi_dims);
}

void
acCommQuit(void)
{
    ERRCHK_MPI_API(MPI_Comm_free(&mpi_comm_));
    ERRCHK_MPI_API(MPI_Finalize());
}

void
acCommGetProcInfo(int* rank, int* nprocs)
{
    *rank   = 0;
    *nprocs = 1;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, rank));
    ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, nprocs));
}

void
print_comm(void)
{
    int rank, nprocs;
    ERRCHK_MPI_API(MPI_Comm_rank(mpi_comm_, &rank));
    ERRCHK_MPI_API(MPI_Comm_size(mpi_comm_, &nprocs));

    int *mpi_dims, *mpi_periods, *mpi_coords;
    ncalloc(as_size_t(mpi_ndims_), mpi_dims);
    ncalloc(as_size_t(mpi_ndims_), mpi_periods);
    ncalloc(as_size_t(mpi_ndims_), mpi_coords);
    ERRCHK_MPI_API(MPI_Cart_get(mpi_comm_, mpi_ndims_, mpi_dims, mpi_periods, mpi_coords));

    for (int i = 0; i < nprocs; ++i) {
        acCommBarrier();
        if (rank == i) {
            printf("Rank %d of %d:\n", rank, nprocs);
            print_array("\t mpi_dims", as_size_t(mpi_ndims_), mpi_dims);
            print_array("\t mpi_periods", as_size_t(mpi_ndims_), mpi_periods);
            print_array("\t mpi_coords", as_size_t(mpi_ndims_), mpi_coords);
        }
        acCommBarrier();
    }
    ndealloc(mpi_dims);
    ndealloc(mpi_periods);
    ndealloc(mpi_coords);
}

void
acCommBarrier(void)
{
    ERRCHK_MPI_API(MPI_Barrier(mpi_comm_));
}

/*
 * HaloSegmentBatch
 */
#include "buf.h"
#include "pack.h"
#include "packet.h"
#include "partition.h"

struct HaloSegmentBatch_s {
    size_t ndims;
    size_t* local_mm;
    size_t* local_nn;
    size_t* local_nn_offset;

    size_t npackets;
    Packet* local_packets;
    Packet* remote_packets;
};

static void halo_segment_batch_verify(const size_t ndims, const size_t* mm, const size_t* nn,
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

    int* coords;
    nalloc(ndims, coords);
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
halo_segment_batch_create(const size_t ndims, const size_t* local_mm, const size_t* local_nn,
                          const size_t* local_nn_offset, const size_t n_aggregate_buffers)
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
        const size_t* subdims = segments.data[i].dims;

        // Recv packet
        const size_t* recv_offset = segments.data[i].offset;
        batch->remote_packets[i]  = packet_create(ndims, subdims, recv_offset, n_aggregate_buffers);

        // Send packet
        size_t* send_offset;
        nalloc(ndims, send_offset);
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
// halo_segment_batch_pack(const size_t ninputs, double** inputs, HaloSegmentBatch* batch)
// {
//     const size_t ndims     = batch->ndims;
//     const size_t* local_mm = batch->local_mm;

//     size_t* zeros;
//     ncalloc(ndims, zeros);

//     for (size_t i = 0; i < batch->npackets; ++i) {
//         for (size_t j = 0; j < ninputs; ++j) {
//             Packet* packet              = &batch->local_packets[i];
//             const size_t* input_dims    = local_mm;
//             const size_t* input_offset  = packet->segment.offset;
//             const double* input         = inputs[j];
//             const size_t* output_dims   = packet->segment.dims;
//             const size_t* output_offset = zeros;
//             double* output              = &packet->buffer
//                                   .data[j * prod(packet->segment.ndims, packet->segment.dims)];
//             pack(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
//         }
//     }
//     ndealloc(zeros);
// }

// static void
// halo_segment_batch_unpack(const HaloSegmentBatch* batch, const size_t noutputs, double** outputs)
// {
//     const size_t ndims     = batch->ndims;
//     const size_t* local_mm = batch->local_mm;

//     size_t* zeros;
//     ncalloc(ndims, zeros);

//     for (size_t i = 0; i < batch->npackets; ++i) {
//         for (size_t j = 0; j < noutputs; ++j) {
//             Packet* packet              = &batch->remote_packets[i];
//             const size_t* output_dims   = local_mm;
//             const size_t* output_offset = packet->segment.offset;
//             double* output              = outputs[j];
//             const size_t* input_dims    = packet->segment.dims;
//             const size_t* input_offset  = zeros;
//             const double* input         = &packet->buffer
//                                        .data[j * prod(packet->segment.ndims,
//                                        packet->segment.dims)];
//             pack(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
//         }
//     }
//     ndealloc(zeros);
// }

void
halo_segment_batch_launch(const size_t ninputs, double* inputs[], struct HaloSegmentBatch_s* batch)
{
    const size_t ndims            = batch->ndims;
    const size_t* local_mm        = batch->local_mm;
    const size_t* local_nn        = batch->local_nn;
    const size_t* local_nn_offset = batch->local_nn_offset;
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

        int* mpi_nn     = to_mpi_format_alloc(ndims, local_nn);
        int* mpi_rr     = to_mpi_format_alloc(ndims, local_nn_offset);
        int* mpi_offset = to_mpi_format_alloc(ndims, remote_packet->segment.offset);
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
halo_segment_batch_wait(struct HaloSegmentBatch_s* batch, const size_t noutputs, double* outputs[])
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
halo_segment_batch_verify(const size_t ndims, const size_t* mm, const size_t* nn,
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
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

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
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

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
    const size_t mm[]        = {8, 8};
    const size_t nn[]        = {6, 6, 6};
    const size_t nn_offset[] = {1, 1, 1};
    const size_t ndims       = ARRAY_SIZE(mm);
    const size_t ninputs     = 3;

    HaloSegmentBatch batch = halo_segment_batch_create(ndims, mm, nn, nn_offset, ninputs);

    double** inputs;
    nalloc(ninputs, inputs);
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

void
test_comm(void)
{
    acCommInit();

    const size_t nn[]  = {8, 8, 8};
    const size_t ndims = ARRAY_SIZE(nn);

    size_t *local_nn, *global_nn_offset;
    nalloc(ndims, local_nn);
    nalloc(ndims, global_nn_offset);

    acCommSetup(ndims, nn, local_nn, global_nn_offset);
    test_halo_segment_batch();

    // print_comm();
    // print_array("local_nn", ndims, local_nn);
    // print_array("global_nn_offset", ndims, global_nn_offset);

    acCommQuit();

    ndealloc(local_nn);
    ndealloc(global_nn_offset);
}
