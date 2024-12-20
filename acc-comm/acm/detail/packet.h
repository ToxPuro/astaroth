#pragma once

#include <memory>

#include "buffer.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

#include "convert.h" // Experimental

template <typename T, typename MemoryResource> class Packet {

  private:
    Shape local_mm;
    Shape local_nn;
    Index local_rr;

    ac::segment segment;

    ac::buffer<T, MemoryResource> send_buffer;
    ac::buffer<T, MemoryResource> recv_buffer;

    MPI_Comm comm{MPI_COMM_NULL};
    MPI_Request send_req{MPI_REQUEST_NULL};
    MPI_Request recv_req{MPI_REQUEST_NULL};

    bool in_progress = false;

  public:
    Packet(const Shape& in_local_mm, const Shape& in_local_nn, const Index& in_local_rr,
           const ac::segment& in_segment, const size_t n_aggregate_buffers)
        : local_mm{in_local_mm},
          local_nn{in_local_nn},
          local_rr{in_local_rr},
          segment{in_segment},
          send_buffer{n_aggregate_buffers * prod(in_segment.dims)},
          recv_buffer{n_aggregate_buffers * prod(in_segment.dims)}
    {
    }

    // Experimental
    ac::mr::base_ptr<T, MemoryResource> get_send_buffer_ptr() { return ac::ptr_cast(send_buffer); }
    ac::mr::base_ptr<T, MemoryResource> get_recv_buffer_ptr() { return ac::ptr_cast(recv_buffer); }

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

    void launch_pipelined(const MPI_Comm& parent_comm,
                          const std::vector<ac::mr::device_ptr<T>>& inputs)
    {
        ERRCHK_MPI(!in_progress);
        in_progress = true;

        // Communicator
        ERRCHK_MPI(comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        // Find the direction and neighbors of the segment
        Index send_offset{((local_nn + segment.offset - local_rr) % local_nn) + local_rr};

        auto recv_direction{ac::mpi::get_direction(segment.offset, local_nn, local_rr)};
        const int recv_neighbor{ac::mpi::get_neighbor(comm, recv_direction)};
        const int send_neighbor{ac::mpi::get_neighbor(comm, -recv_direction)};

        // Calculate the bytes to send
        const size_t count{inputs.size() * prod(segment.dims)};
        ERRCHK_MPI(count <= send_buffer.size());
        ERRCHK_MPI(count <= recv_buffer.size());

        // Post recv
        const int tag{0};
        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 recv_neighbor,
                                 tag,
                                 comm,
                                 &recv_req));

        // Pack and post send
        pack(local_mm,
             segment.dims,
             send_offset,
             inputs,
             ac::mr::device_ptr<T>{send_buffer.size(), send_buffer.data()});

        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Isend(send_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 send_neighbor,
                                 tag,
                                 comm,
                                 &send_req));
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

    void wait(std::vector<ac::mr::device_ptr<T>>& outputs)
    {
        ERRCHK_MPI(in_progress);
        ERRCHK_MPI_API(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));
        unpack(ac::mr::device_ptr<T>{recv_buffer.size(), recv_buffer.data()},
               local_mm,
               segment.dims,
               segment.offset,
               outputs);

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

    // Experimental
    void launch_batched(const MPI_Comm& parent_comm)
    {
        ERRCHK_MPI(!in_progress);
        in_progress = true;

        // Communicator
        ERRCHK_MPI(comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &comm));

        // Find the direction and neighbors of the segment
        Index send_offset{((local_nn + segment.offset - local_rr) % local_nn) + local_rr};

        auto recv_direction{ac::mpi::get_direction(segment.offset, local_nn, local_rr)};
        const int recv_neighbor{ac::mpi::get_neighbor(comm, recv_direction)};
        const int send_neighbor{ac::mpi::get_neighbor(comm, -recv_direction)};

        // Calculate the bytes to send
        // const size_t count{inputs.size() * prod(segment.dims)};
        // ERRCHK_MPI(count <= send_buffer.size());
        // ERRCHK_MPI(count <= recv_buffer.size());
        const size_t count{send_buffer.size()};

        // Post recv
        const int tag{0};
        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Irecv(recv_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 recv_neighbor,
                                 tag,
                                 comm,
                                 &recv_req));

        // Post send

        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Isend(send_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 send_neighbor,
                                 tag,
                                 comm,
                                 &send_req));
    }

    void wait_batched()
    {
        ERRCHK_MPI(in_progress);
        ERRCHK_MPI_API(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));

        ERRCHK_MPI_API(MPI_Wait(&send_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Comm_free(&comm));

        // Check that the MPI implementation reset the resources
        ERRCHK_MPI(recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(comm == MPI_COMM_NULL);

        // Complete
        in_progress = false;
    }

    friend std::ostream& operator<<(std::ostream& os, const Packet<T, MemoryResource>& obj)
    {
        os << "{";
        os << "segment: " << obj.segment << ", ";
        os << "buffer: " << obj.buffer << ", ";
        os << "req: " << obj.req;
        os << "}";
        return os;
    }

    Packet(const Packet&)            = delete; // Copy constructor
    Packet& operator=(const Packet&) = delete; // Copy assignment operator
    Packet(Packet&&)                 = delete; // Move constructor
    Packet& operator=(Packet&&)      = delete; // Move assignment operator
};

void test_packet();
