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
 * HaloSegmentBatch
 */
#include "buf.h"
#include "partition.h"

#define COMM_DATATYPE (MPI_UNSIGNED_LONG_LONG)

typedef struct {
    Segment segment;
    Buffer buffer;
    MPI_Request req;
    MPI_Datatype subarray;
} Packet;

static Packet
packet_create(const size_t ndims, const size_t* local_mm, const size_t* subdims,
              const size_t* offset, const size_t nbuffers)
{
    Packet packet;
    packet.segment = segment_create(ndims, subdims, offset);
    packet.buffer  = buffer_create(prod(ndims, subdims) * nbuffers);
    packet.req     = MPI_REQUEST_NULL;

    int *mpi_dims, *mpi_subdims, *mpi_offset;
    ncalloc(ndims, mpi_dims);
    ncalloc(ndims, mpi_subdims);
    ncalloc(ndims, mpi_offset);
    to_mpi_format(ndims, local_mm, mpi_dims);
    to_mpi_format(ndims, subdims, mpi_subdims);
    to_mpi_format(ndims, offset, mpi_offset);
    ERRCHK_MPI_API(MPI_Type_create_subarray(as_int(ndims), mpi_dims, mpi_subdims, mpi_offset,
                                            MPI_ORDER_C, COMM_DATATYPE, &packet.subarray));
    ERRCHK_MPI_API(MPI_Type_commit(&packet.subarray));
    ndealloc(mpi_dims);
    ndealloc(mpi_subdims);
    ndealloc(mpi_offset);

    return packet;
}

static void
packet_destroy(Packet* packet)
{
    ERRCHK_MPI_API(MPI_Type_free(&packet->subarray));
    ERRCHK_MPI_API(MPI_Request_free(&packet->req))
    buffer_destroy(&packet->buffer);
    segment_destroy(&packet->segment);
}

static void
print_packet(const char* label, const Packet packet)
{
    printf("%s:\n", label);
    print_segment("", packet.segment);
}

typedef dynarr_s(Packet) PacketArray;

typedef struct {
    PacketArray send_packets;
    PacketArray recv_packets;
} HaloSegmentBatch;

HaloSegmentBatch
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
    const size_t nbuffers = 1;
    WARNING("using hardcoded nbuffers = 1");

    HaloSegmentBatch batch;
    dynarr_create_with_destructor(packet_destroy, &batch.send_packets);
    dynarr_create_with_destructor(packet_destroy, &batch.recv_packets);

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
        dynarr_append(packet_create(ndims, local_mm, subdims, recv_offset, nbuffers),
                      &batch.recv_packets);

        // Send packet
        size_t* send_offset;
        nalloc(ndims, send_offset);
        for (size_t j = 0; j < ndims; ++j)
            send_offset[j] = ((local_nn[j] + recv_offset[j] - local_nn_offset[j]) % local_nn[j]) +
                             local_nn_offset[j];
        dynarr_append(packet_create(ndims, local_mm, subdims, send_offset, nbuffers),
                      &batch.send_packets);

        ndealloc(send_offset);
    }

    // Cleanup
    dynarr_destroy(&segments);

    return batch;
}

void
halo_segment_batch_destroy(HaloSegmentBatch* batch)
{
    dynarr_destroy(&batch->send_packets);
    dynarr_destroy(&batch->recv_packets);
}

void
test_halo_segment_batch(void)
{
    const size_t mm[]        = {8, 8};
    const size_t nn[]        = {6, 6, 6};
    const size_t nn_offset[] = {1, 1, 1};
    const size_t ndims       = ARRAY_SIZE(mm);

    HaloSegmentBatch batch = halo_segment_batch_create(ndims, mm, nn, nn_offset, 1);

    printf("Created halo segments\n");
    print("batch.send_packets.length", batch.send_packets.length);
    for (size_t i = 0; i < batch.send_packets.length; ++i) {
        printf("%zu:\n", i);
        print_packet("Send", batch.send_packets.data[i]);
        print_packet("Recv", batch.recv_packets.data[i]);
    }

    halo_segment_batch_destroy(&batch);
}

/*
 * Comm
 */
static MPI_Comm mpi_comm_ = MPI_COMM_NULL;
static int mpi_ndims_     = -1;

void
acCommInit(const size_t ndims, const size_t* global_nn, size_t* local_nn, size_t* global_nn_offset)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));

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

void
test_comm(void)
{
    const size_t nn[]  = {8, 8, 8};
    const size_t ndims = ARRAY_SIZE(nn);

    size_t *local_nn, *global_nn_offset;
    nalloc(ndims, local_nn);
    nalloc(ndims, global_nn_offset);

    acCommInit(ndims, nn, local_nn, global_nn_offset);
    test_halo_segment_batch();

    print_comm();
    print_array("local_nn", ndims, local_nn);
    print_array("global_nn_offset", ndims, global_nn_offset);

    acCommQuit();

    ndealloc(local_nn);
    ndealloc(global_nn_offset);
}
