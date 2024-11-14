#pragma once

#include <memory>

#include "buffer.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

template <typename T> class Packet {

  private:
    Shape local_mm;
    Shape local_nn;
    Index local_rr;

    Segment segment; // Shape of the data block the packet represents (for packing or unpacking)

    Buffer<T, DeviceMemoryResource> send_buffer;
    Buffer<T, DeviceMemoryResource> recv_buffer;

    MPI_Comm comm{MPI_COMM_NULL};
    MPI_Request send_req{MPI_REQUEST_NULL};
    MPI_Request recv_req{MPI_REQUEST_NULL};

    bool in_progress = false;

  public:
    Packet(const Shape& in_local_mm, const Shape& in_local_nn, const Index& in_local_rr,
           const Segment& in_segment, const size_t n_aggregate_buffers)
        : local_mm(in_local_mm),
          local_nn(in_local_nn),
          local_rr(in_local_rr),
          segment(in_segment),
          send_buffer(n_aggregate_buffers * prod(in_segment.dims)),
          recv_buffer(n_aggregate_buffers * prod(in_segment.dims))
    {
    }

    ~Packet()
    {
        ERRCHK_MPI(!in_progress);

        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        if (recv_req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&recv_req));

        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        if (send_req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&send_req));

        ERRCHK_MPI(comm == MPI_COMM_NULL);
        if (comm != MPI_COMM_NULL)
            ERRCHK_MPI_API(MPI_Comm_free(&comm));
    }

    Packet(const Packet&)            = delete; // Copy constructor
    Packet& operator=(const Packet&) = delete; // Copy assignment operator
    Packet(Packet&&)                 = delete; // Move constructor
    Packet& operator=(Packet&&)      = delete; // Move assignment operator

    void launch(const MPI_Comm& parent_comm, const PackPtrArray<T*>& inputs)
    {
        ERRCHK_MPI(!in_progress);
        in_progress = true;

        // Communicator
        ERRCHK_MPI(comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        // Find the direction and neighbors of the segment
        const size_t count = inputs.count * prod(segment.dims);
        ERRCHK_MPI(count <= send_buffer.size());
        ERRCHK_MPI(count <= recv_buffer.size());

        Index send_offset = ((local_nn + segment.offset - local_rr) % local_nn) + local_rr;

        const Direction recv_direction = get_direction(segment.offset, local_nn, local_rr);
        const int recv_neighbor        = get_neighbor(comm, recv_direction);
        const int send_neighbor        = get_neighbor(comm, -recv_direction);

        // Post recv
        const int tag = 0;
        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data(), as<int>(count), get_mpi_dtype<T>(),
                                 recv_neighbor, tag, comm, &recv_req));

        // Pack and post send
        pack(local_mm, segment.dims, send_offset, inputs, send_buffer);

        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Isend(send_buffer.data(), as<int>(count), get_mpi_dtype<T>(),
                                 send_neighbor, tag, comm, &send_req));
    }

    bool ready() const
    {
        ERRCHK_MPI(in_progress);
        ERRCHK_MPI(send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(recv_req != MPI_REQUEST_NULL);

        int send_flag, recv_flag;
        ERRCHK_MPI_API(MPI_Request_get_status(send_req, &send_flag, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Request_get_status(recv_req, &recv_flag, MPI_STATUS_IGNORE));
        return in_progress && send_flag && recv_flag;
    };

    void wait(PackPtrArray<T*>& outputs)
    {
        ERRCHK_MPI(in_progress);
        ERRCHK_MPI_API(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));
        unpack(recv_buffer, local_mm, segment.dims, segment.offset, outputs);

        ERRCHK_MPI_API(MPI_Wait(&send_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Comm_free(&comm));

        // Check that the MPI implementation reset the resources
        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(comm == MPI_COMM_NULL);

        // Complete
        in_progress = false;
    }

    bool complete() const { return !in_progress; };

    friend __host__ std::ostream& operator<<(std::ostream& os, const Packet<T>& obj)
    {
        os << "{";
        os << "segment: " << obj.segment << ", ";
        os << "buffer: " << obj.buffer << ", ";
        os << "req: " << obj.req;
        os << "}";
        return os;
    }
};

void test_packet();
