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

/*
 * Comm
 */
static MPI_Comm mpi_comm_ = MPI_COMM_NULL;
static int mpi_ndims_     = -1;

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
    ndealloc(coords);

    ndealloc(mpi_coords);

    // Cleanup
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
#include "partition.h"

#define COMM_DATATYPE (MPI_UNSIGNED_LONG_LONG)

typedef struct {
    Segment segment;
    Buffer buffer;
    MPI_Request req;
} Packet;

static Packet
packet_create(const size_t ndims, const size_t* dims, const size_t* offset, const size_t nbuffers)
{
    Packet packet;
    packet.segment = segment_create(ndims, dims, offset);
    packet.buffer  = buffer_create(prod(ndims, dims) * nbuffers);
    packet.req     = MPI_REQUEST_NULL;

    return packet;
}

static void
packet_wait(Packet* packet)
{
    if (packet->req != MPI_REQUEST_NULL) {
        MPI_Status status;
        ERRCHK_MPI_API(MPI_Wait(&packet->req, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        ERRCHK_MPI_API(MPI_Request_free(&packet->req));
    }
    else {
        WARNING("packet_wait called but no there is packet to wait for");
    }
}

static void
packet_destroy(Packet* packet)
{
    if (packet->req != MPI_REQUEST_NULL)
        packet_wait(packet);
    ERRCHK(packet->req == MPI_REQUEST_NULL); // Confirm that the request is deallocated
    buffer_destroy(&packet->buffer);
    segment_destroy(&packet->segment);
}

struct HaloSegmentBatch_s {
    size_t ndims;
    size_t* local_mm;

    size_t npackets;
    Packet* local_packets;
    Packet* remote_packets;
};

static void halo_segment_batch_verify(const size_t ndims, const size_t* mm, const size_t* nn,
                                      const struct HaloSegmentBatch_s batch);

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

void
halo_segment_batch_wait(struct HaloSegmentBatch_s* batch)
{
    for (size_t i = 0; i < batch->npackets; ++i) {
        packet_wait(&batch->local_packets[i]);
        packet_wait(&batch->remote_packets[i]);
    }
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
halo_segment_batch_launch(const size_t ninputs, const double* inputs[],
                          struct HaloSegmentBatch_s* batch)
{
    for (size_t i = 0; i < batch->npackets; ++i) {
        Packet* packet = &batch->local_packets[i];
        pack(batch->ndims, batch->local_mm, packet->segment.dims, packet->segment.offset, ninputs,
             inputs, packet->buffer.data);
        // TODO CONTINUE HERE
        // Post recv (asynchronous)
        // Pack packet (synchronous)
        // Send packet (asynchronous)
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
