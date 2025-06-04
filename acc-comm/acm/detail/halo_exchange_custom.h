#pragma once

#include <mpi.h>
#include <vector>

#include "buffer.h"
#include "pack.h"
#include "partition.h"
#include "type_conversion.h"
#include "view.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

namespace acm {

template <typename T, typename Allocator> class packet {
  private:
    ac::shape m_local_mm;
    ac::shape m_local_nn;
    ac::index m_local_rr;

    ac::segment m_recv_segment;
    ac::segment m_send_segment;

    ac::buffer<T, Allocator> m_recv_buffer;
    ac::buffer<T, Allocator> m_send_buffer;

    MPI_Comm m_comm{MPI_COMM_NULL};
    int16_t  m_tag{0};

    int m_recv_neighbor{MPI_PROC_NULL};
    int m_send_neighbor{MPI_PROC_NULL};

    MPI_Request m_recv_req{MPI_REQUEST_NULL};
    MPI_Request m_send_req{MPI_REQUEST_NULL};

  public:
    packet(const MPI_Comm& parent_comm, const ac::shape& local_mm, const ac::shape& local_nn,
           const ac::index& local_rr, const ac::segment& recv_segment,
           const size_t n_max_aggregate_buffers)
        : m_local_mm{local_mm},
          m_local_nn{local_nn},
          m_local_rr{local_rr},
          m_recv_segment{recv_segment},
          m_send_segment{recv_segment.dims,
                         ((local_nn + recv_segment.offset - local_rr) % local_nn) + local_rr},
          m_recv_buffer{n_max_aggregate_buffers * prod(recv_segment.dims)},
          m_send_buffer{n_max_aggregate_buffers * prod(recv_segment.dims)}
    {
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &m_comm));

        const ac::direction recv_direction{
            ac::mpi::get_direction(m_recv_segment.offset, m_local_nn, m_local_rr)};
        m_recv_neighbor = ac::mpi::get_neighbor(m_comm, recv_direction);
        m_send_neighbor = ac::mpi::get_neighbor(m_comm, -recv_direction);
    }

    ~packet()
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Comm_free(&m_comm));
    }

    packet(const packet&)            = delete; // Copy constructor
    packet& operator=(const packet&) = delete; // Copy assignment operator
    packet(packet&&)                 = delete; // Move constructor
    packet& operator=(packet&&)      = delete; // Move assignment operator

    void launch(const std::vector<ac::view<T, Allocator>>& inputs)
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);

        // Calculate the number of elements to send
        const size_t count{inputs.size() * prod(m_recv_segment.dims)};
        ERRCHK_MPI(count <= m_send_buffer.size());
        ERRCHK_MPI(count <= m_recv_buffer.size());

        // Pack directly to the recv buffer if the destination is on the same device
        const auto rank{ac::mpi::get_rank(m_comm)};
        if (rank == m_recv_neighbor && rank == m_send_neighbor) {
            acm::pack(m_local_mm,
                      m_send_segment.dims,
                      m_send_segment.offset,
                      inputs,
                      m_recv_buffer.get());

            // Create a dummy request to simplify asynchronous management
            static int dummy{-1};
            ERRCHK_MPI_API(MPI_Irecv(&dummy, 0, MPI_INT, rank, m_tag, m_comm, &m_recv_req));
            ERRCHK_MPI_API(MPI_Isend(&dummy, 0, MPI_INT, rank, m_tag, m_comm, &m_send_req));
        }
        else { // Use MPI communication

            // Post recv
            ERRCHK_MPI_API(MPI_Irecv(m_recv_buffer.data(),
                                     as<int>(count),
                                     ac::mpi::get_dtype<T>(),
                                     m_recv_neighbor,
                                     m_tag,
                                     m_comm,
                                     &m_recv_req));

            // Pack and post send
            acm::pack(m_local_mm,
                      m_send_segment.dims,
                      m_send_segment.offset,
                      inputs,
                      m_send_buffer.get());

            ERRCHK_MPI_API(MPI_Isend(m_send_buffer.data(),
                                     as<int>(count),
                                     ac::mpi::get_dtype<T>(),
                                     m_send_neighbor,
                                     m_tag,
                                     m_comm,
                                     &m_send_req));
        }

        ac::mpi::increment_tag(m_tag);
    }

    /** Completes the task */
    void wait(std::vector<ac::view<T, Allocator>> outputs)
    {
        ERRCHK_MPI(m_send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req != MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Wait(&m_recv_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Wait(&m_send_req, MPI_STATUS_IGNORE));
        acm::unpack(m_recv_buffer.get(),
                    m_local_mm,
                    m_recv_segment.dims,
                    m_recv_segment.offset,
                    outputs);

        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);
    }

    /** Returns true if the task is ready to be waited on */
    bool ready() const
    {
        ERRCHK_MPI(m_send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req != MPI_REQUEST_NULL);

        int send_flag, recv_flag;
        ERRCHK_MPI_API(MPI_Request_get_status(m_send_req, &send_flag, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Request_get_status(m_recv_req, &recv_flag, MPI_STATUS_IGNORE));
        return send_flag && recv_flag;
    };

    /** Returns true if there is no task in flight */
    bool complete() const
    {
        return (m_send_req == MPI_REQUEST_NULL) && (m_recv_req == MPI_REQUEST_NULL);
    };
};

template <typename T, typename Allocator> class halo_exchange {
  private:
    std::vector<std::unique_ptr<packet<T, Allocator>>> m_packets;

  public:
    halo_exchange() = default;

    halo_exchange(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
                  const size_t n_max_aggregate_buffers)
    {
        // Get local mesh dimensions
        const auto local_mm{ac::mpi::get_local_mm(parent_comm, global_nn, rr)};
        const auto local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};

        // Partition the domain
        auto segments{partition(local_mm, local_nn, rr)};

        // Prune the segment containing the computational domain
        for (size_t i{0}; i < segments.size(); ++i) {
            if (within_box(segments[i].offset, local_nn, rr)) {
                segments.erase(segments.begin() + as<long>(i));
                --i;
            }
        }

        // Sort the packets from largest to smallest
        std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
            return ac::prod(a.dims) > ac::prod(b.dims);
        });

        for (const auto& segment : segments)
            m_packets.push_back(std::make_unique<packet<T, Allocator>>(parent_comm,
                                                                       local_mm,
                                                                       local_nn,
                                                                       rr,
                                                                       segment,
                                                                       n_max_aggregate_buffers));
    }

    void launch(const std::vector<ac::view<T, Allocator>>& inputs)
    {
        ERRCHK_MPI(complete());

        for (auto& packet : m_packets)
            packet->launch(inputs);
    }

    /** Waits one message and returns */
    void wait_one(std::vector<ac::view<T, Allocator>> outputs)
    {
        ERRCHK_MPI(!complete());

        for (auto& packet : m_packets)
            if (!packet->complete() && packet->ready())
                packet->wait(outputs);
    }

    /** Waits all message and returns */
    void wait(std::vector<ac::view<T, Allocator>> outputs)
    {
        ERRCHK_MPI(!complete());

        while (!complete())
            wait_one(outputs);
    }

    /** Returns true if there is at least one message that can be waited on */
    bool ready() const
    {
        ERRCHK_MPI(!complete());

        for (const auto& packet : m_packets)
            if (!packet->complete() && packet->ready())
                return true;

        return false;
    }

    /** Returns true if all messages have been completed */
    bool complete() const
    {
        for (const auto& packet : m_packets)
            if (!packet->complete())
                return false;

        return true;
    }
};

} // namespace acm
