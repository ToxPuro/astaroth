#pragma once

#include <memory>

#include "buffer.h"
#include "pack.h"
#include "segment.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

#include <mpi.h>

#include "type_conversion.h"

namespace ac::comm {

template <typename T, typename Allocator> class packet {

  private:
    ac::shape m_local_mm;
    ac::shape m_local_nn;
    ac::index m_local_rr;

    ac::segment m_segment;

    ac::buffer<T, Allocator> m_send_buffer;
    ac::buffer<T, Allocator> m_recv_buffer;

    MPI_Comm    m_comm{MPI_COMM_NULL};
    MPI_Request m_send_req{MPI_REQUEST_NULL};
    MPI_Request m_recv_req{MPI_REQUEST_NULL};

    bool m_in_progress = false;

  public:
    packet(const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
           const ac::segment& segment, const size_t n_aggregate_buffers)
        : m_local_mm{local_mm},
          m_local_nn{local_nn},
          m_local_rr{local_rr},
          m_segment{segment},
          m_send_buffer{n_aggregate_buffers * prod(segment.dims)},
          m_recv_buffer{n_aggregate_buffers * prod(segment.dims)}
    {
    }

    ~packet()
    {
        ERRCHK_MPI(!m_in_progress);

        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
        if (m_recv_req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&m_recv_req));

        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        if (m_send_req != MPI_REQUEST_NULL)
            ERRCHK_MPI_API(MPI_Request_free(&m_send_req));

        ERRCHK_MPI(m_comm == MPI_COMM_NULL);
        if (m_comm != MPI_COMM_NULL)
            ERRCHK_MPI_API(MPI_Comm_free(&m_comm));
    }

    void launch(const MPI_Comm& parent_comm, const std::vector<ac::mr::device_pointer<T>>& inputs)
    {
        ERRCHK_MPI(!m_in_progress);
        m_in_progress = true;

        // Communicator
        ERRCHK_MPI(m_comm == MPI_COMM_NULL);
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &m_comm));

        // Find the direction and neighbors of the segment
        ac::index send_offset{((m_local_nn + m_segment.offset - m_local_rr) % m_local_nn) +
                              m_local_rr};

        auto      recv_direction{ac::mpi::get_direction(m_segment.offset, m_local_nn, m_local_rr)};
        const int recv_neighbor{ac::mpi::get_neighbor(m_comm, recv_direction)};
        const int send_neighbor{ac::mpi::get_neighbor(m_comm, -recv_direction)};

        // Calculate the bytes to send
        const size_t count{inputs.size() * prod(m_segment.dims)};
        ERRCHK_MPI(count <= m_send_buffer.size());
        ERRCHK_MPI(count <= m_recv_buffer.size());

        // Post recv
        const int tag{0};
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Irecv(m_recv_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 recv_neighbor,
                                 tag,
                                 m_comm,
                                 &m_recv_req));

        // Pack and post send
        pack(m_local_mm, m_segment.dims, send_offset, inputs, m_send_buffer.get());

        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Isend(m_send_buffer.data(),
                                 as<int>(count),
                                 ac::mpi::get_dtype<T>(),
                                 send_neighbor,
                                 tag,
                                 m_comm,
                                 &m_send_req));
    }

    bool ready() const
    {
        ERRCHK_MPI(m_in_progress);
        ERRCHK_MPI(m_send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req != MPI_REQUEST_NULL);

        int send_flag, recv_flag;
        ERRCHK_MPI_API(MPI_Request_get_status(m_send_req, &send_flag, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Request_get_status(m_recv_req, &recv_flag, MPI_STATUS_IGNORE));
        return m_in_progress && send_flag && recv_flag;
    };

    void wait(std::vector<ac::mr::device_pointer<T>>& outputs)
    {
        ERRCHK_MPI(m_in_progress);
        ERRCHK_MPI_API(MPI_Wait(&m_recv_req, MPI_STATUS_IGNORE));
        unpack(m_recv_buffer.get(), m_local_mm, m_segment.dims, m_segment.offset, outputs);

        ERRCHK_MPI_API(MPI_Wait(&m_send_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Comm_free(&m_comm));

        // Check that the MPI implementation reset the resources
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_comm == MPI_COMM_NULL);

        // Complete
        m_in_progress = false;
    }

    bool complete() const { return !m_in_progress; };

    friend std::ostream& operator<<(std::ostream& os, const packet<T, Allocator>& obj)
    {
        os << "{";
        os << "segment: " << obj.m_segment << ", ";
        os << "buffer: " << obj.buffer << ", ";
        os << "req: " << obj.req;
        os << "}";
        return os;
    }

    packet(const packet&)            = delete; // Copy constructor
    packet& operator=(const packet&) = delete; // Copy assignment operator
    packet(packet&&)                 = delete; // Move constructor
    packet& operator=(packet&&)      = delete; // Move assignment operator
};

} // namespace ac::comm

void test_packet();
