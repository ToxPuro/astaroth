#pragma once

#include <mpi.h>

#include "acm/detail/buffer.h"
#include "acm/detail/experimental/mpi_utils_experimental.h"

namespace ac::mpi::hindexed {

template <typename T, typename Allocator> class packet {
  private:
    ac::mpi::comm              m_comm;
    ac::mpi::hindexed_block<T> m_send_block;
    ac::mpi::hindexed_block<T> m_recv_block;

    ac::mpi::request m_send_req;
    ac::mpi::request m_recv_req;
    int16_t          m_tag{0};

    int m_send_neighbor{MPI_PROC_NULL};
    int m_recv_neighbor{MPI_PROC_NULL};

  public:
    packet(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
           const ac::segment& segment, const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
           const std::vector<ac::mr::pointer<T, Allocator>>& outputs)
        : m_comm{parent_comm}
    {
        ERRCHK_MPI(inputs.size() == outputs.size());

        const auto local_mm{ac::mpi::get_local_mm(m_comm.get(), global_nn, rr)};
        const auto local_nn{ac::mpi::get_local_nn(m_comm.get(), global_nn)};
        const auto recv_offset{segment.offset};
        const auto send_offset{((local_nn + recv_offset - rr) % local_nn) + rr};

        m_recv_block = ac::mpi::hindexed_block<T>{local_mm,
                                                  segment.dims,
                                                  recv_offset,
                                                  ac::unwrap_data(outputs)};
        m_send_block = ac::mpi::hindexed_block<T>{local_mm,
                                                  segment.dims,
                                                  send_offset,
                                                  ac::unwrap_data(inputs)};

        const ac::direction recv_direction{ac::mpi::get_direction(segment.offset, local_nn, rr)};
        m_recv_neighbor = ac::mpi::get_neighbor(m_comm.get(), recv_direction);
        m_send_neighbor = ac::mpi::get_neighbor(m_comm.get(), -recv_direction);
    }

    void launch()
    {
        ERRCHK_MPI_API(MPI_Irecv(MPI_BOTTOM,
                                 1,
                                 m_recv_block.get(),
                                 m_recv_neighbor,
                                 m_tag,
                                 m_comm.get(),
                                 m_recv_req.data()));
        ERRCHK_MPI_API(MPI_Isend(MPI_BOTTOM,
                                 1,
                                 m_send_block.get(),
                                 m_send_neighbor,
                                 m_tag,
                                 m_comm.get(),
                                 m_send_req.data()));

        ac::mpi::increment_tag(m_tag);
    }

    /** Completes the task */
    void wait()
    {
        m_send_req.wait();
        m_recv_req.wait();
    }

    /** Returns true if the task is ready to be waited on */
    bool ready() const { return m_send_req.ready() && m_recv_req.ready(); };

    /** Returns true if there is no task in flight */
    bool complete() const { return m_send_req.complete() && m_recv_req.complete(); };
};

template <typename T, typename Allocator> class halo_exchange {
  private:
    std::vector<packet<T, Allocator>> m_packets;

  public:
    halo_exchange(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
                  const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                  const std::vector<ac::mr::pointer<T, Allocator>>& outputs)
    {
        const auto mm{ac::mpi::get_local_mm(parent_comm, global_nn, rr)};
        const auto nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
        const auto segments{prune(partition(mm, nn, rr), nn, rr)};
        for (const auto& segment : segments)
            m_packets.push_back({parent_comm, global_nn, rr, segment, inputs, outputs});
    }

    void launch()
    {
        for (auto& packet : m_packets)
            packet.launch();
    }

    /** Waits one message and returns */
    void wait_one()
    {
        ERRCHK_MPI(!complete());

        for (auto& packet : m_packets)
            if (!packet.complete() && packet.ready())
                packet.wait();
    }

    /** Waits all message and returns */
    void wait_all()
    {
        ERRCHK_MPI(!complete());

        while (!complete())
            wait_one();
    }

    /** Returns true if there is at least one message that can be waited on */
    bool ready() const
    {
        ERRCHK_MPI(!complete());

        for (const auto& packet : m_packets)
            if (!packet.complete() && packet.ready())
                return true;

        return false;
    }

    /** Returns true if all messages have been completed */
    bool complete() const
    {
        for (const auto& packet : m_packets)
            if (!packet.complete())
                return false;

        return true;
    }
};

} // namespace ac::mpi::hindexed
