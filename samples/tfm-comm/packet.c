#include "packet.h"

#include "errchk_mpi.h"
#include "math_utils.h"

Packet
packet_create(const size_t ndims, const size_t* dims, const size_t* offset, const size_t nbuffers)
{
    Packet packet;

    packet.segment = segment_create(ndims, dims, offset);
    packet.buffer  = buffer_create(nbuffers * prod(ndims, dims));
    packet.req     = MPI_REQUEST_NULL;

    return packet;
}

void
packet_wait(Packet* packet)
{
    if (packet->req != MPI_REQUEST_NULL) {
        // Note: MPI_Status needs to be initialized.
        // Otherwise leads to uninitialized memory access and causes spurious errors
        // because MPI_Wait does not modify the status on successful MPI_Wait
        MPI_Status status = {.MPI_ERROR = MPI_SUCCESS};
        ERRCHK_MPI_API(MPI_Wait(&packet->req, &status));
        ERRCHK_MPI_API(status.MPI_ERROR);
        // Some MPI implementations free the request with MPI_Wait
        if (packet->req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&packet->req));
    }
    else {
        WARNING("packet_wait called but no there is packet to wait for");
    }
}

void
packet_destroy(Packet* packet)
{
    if (packet->req != MPI_REQUEST_NULL)
        packet_wait(packet);
    ERRCHK(packet->req == MPI_REQUEST_NULL); // Confirm that the request is deallocated
    buffer_destroy(&packet->buffer);
    segment_destroy(&packet->segment);
}
