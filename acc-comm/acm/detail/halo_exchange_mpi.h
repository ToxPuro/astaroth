#pragma once

#include <mpi.h>
#include <vector>

#include "buffer.h"
#include "partition.h"
#include "pointer.h"
#include "type_conversion.h"

#include "errchk_mpi.h"
#include "mpi_utils.h"

namespace ac::mpi {

template <typename T, typename Allocator> class packet {
  private:
    MPI_Comm m_comm{MPI_COMM_NULL};
    int16_t  m_tag{0};

    MPI_Datatype m_recv_subarray{MPI_DATATYPE_NULL};
    MPI_Datatype m_send_subarray{MPI_DATATYPE_NULL};

    MPI_Request m_recv_req{MPI_REQUEST_NULL};
    MPI_Request m_send_req{MPI_REQUEST_NULL};

    int m_recv_neighbor{MPI_PROC_NULL};
    int m_send_neighbor{MPI_PROC_NULL};

  public:
    packet(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
           const ac::segment& segment)
    {
        ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, &m_comm));
        const auto local_mm{ac::mpi::get_local_mm(m_comm, global_nn, rr)};
        const auto local_nn{ac::mpi::get_local_nn(m_comm, global_nn)};

        const auto recv_offset{segment.offset};
        const auto send_offset{((local_nn + recv_offset - rr) % local_nn) + rr};

        m_recv_subarray = ac::mpi::subarray_create(local_mm,
                                                   segment.dims,
                                                   recv_offset,
                                                   ac::mpi::get_dtype<T>());
        m_send_subarray = ac::mpi::subarray_create(local_mm,
                                                   segment.dims,
                                                   send_offset,
                                                   ac::mpi::get_dtype<T>());

        const ac::direction recv_direction{ac::mpi::get_direction(segment.offset, local_nn, rr)};
        m_recv_neighbor = ac::mpi::get_neighbor(m_comm, recv_direction);
        m_send_neighbor = ac::mpi::get_neighbor(m_comm, -recv_direction);
    }

    ~packet()
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Type_free(&m_send_subarray));
        ERRCHK_MPI_API(MPI_Type_free(&m_recv_subarray));

        ERRCHK_MPI_API(MPI_Comm_free(&m_comm));
    }

    packet(const packet&)            = delete; // Copy constructor
    packet& operator=(const packet&) = delete; // Copy assignment operator
    packet(packet&&)                 = delete; // Move constructor
    packet& operator=(packet&&)      = delete; // Move assignment operator

    void launch(const ac::mr::pointer<T, Allocator>& input, ac::mr::pointer<T, Allocator> output)
    {
        ERRCHK_MPI(m_send_req == MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req == MPI_REQUEST_NULL);

        // Should bypass MPI when communicating to self:
        //  Notes: MPI_Pack is synchronous and inefficient for moving device memory
        // (possibly the reason why MPI communication to self is relatively slow),
        //  should use custom pack/unpack kernels instead in the same way as in
        //  `halo_exchange_packed.h`.
        //
        //
        // const auto rank{ac::mpi::get_rank(m_comm)};
        // if (rank == m_recv_neighbor && rank == m_send_neighbor) {
        //     // int recv_pack_bytes{-1};
        //     // int send_pack_bytes{-1};
        //     // ERRCHK_MPI_API(MPI_Pack_size(1, m_recv_subarray, m_comm, &recv_pack_bytes));
        //     // ERRCHK_MPI_API(MPI_Pack_size(1, m_send_subarray, m_comm, &send_pack_bytes));
        //     // ERRCHK_MPI(recv_pack_bytes == send_pack_bytes);

        //     // int                             position{0};
        //     // const size_t                    count{as<size_t>(recv_pack_bytes) / sizeof(T)};
        //     // static ac::buffer<T, Allocator> tmp{count};
        //     // ERRCHK_MPI_API(MPI_Pack(input.data(),
        //     //                         1,
        //     //                         m_send_subarray,
        //     //                         tmp.data(),
        //     //                         as<int>(recv_pack_bytes),
        //     //                         &position,
        //     //                         m_comm));
        //     // ERRCHK_MPI_API(MPI_Unpack(tmp.data(),
        //     //                           as<int>(recv_pack_bytes),
        //     //                           &position,
        //     //                           output.data(),
        //     //                           1,
        //     //                           m_recv_subarray,
        //     //                           m_comm));

        //     // Dummy request
        //     static int dummy{-1};
        //     ERRCHK_MPI_API(MPI_Isend(&dummy, 0, MPI_INT, rank, 0, m_comm, &m_send_req));
        //     ERRCHK_MPI_API(MPI_Irecv(&dummy, 0, MPI_INT, rank, 0, m_comm, &m_recv_req));
        // }
        // else {
        ERRCHK_MPI_API(MPI_Irecv(output.data(),
                                 1,
                                 m_recv_subarray,
                                 m_recv_neighbor,
                                 m_tag,
                                 m_comm,
                                 &m_recv_req));
        ERRCHK_MPI_API(MPI_Isend(input.data(),
                                 1,
                                 m_send_subarray,
                                 m_send_neighbor,
                                 m_tag,
                                 m_comm,
                                 &m_send_req));
        // }

        ac::mpi::increment_tag(m_tag);
    }

    /** Completes the task */
    void wait()
    {
        ERRCHK_MPI(m_send_req != MPI_REQUEST_NULL);
        ERRCHK_MPI(m_recv_req != MPI_REQUEST_NULL);

        ERRCHK_MPI_API(MPI_Wait(&m_send_req, MPI_STATUS_IGNORE));
        ERRCHK_MPI_API(MPI_Wait(&m_recv_req, MPI_STATUS_IGNORE));

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
    halo_exchange(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr)
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
            m_packets.push_back(
                std::make_unique<packet<T, Allocator>>(parent_comm, global_nn, rr, segment));
    }

    void launch(const ac::mr::pointer<T, Allocator>& input, ac::mr::pointer<T, Allocator> output)
    {
        ERRCHK_MPI(complete());

        for (auto& packet : m_packets)
            packet->launch(input, output);
    }

    /** Waits one message and returns */
    void wait_one()
    {
        ERRCHK_MPI(!complete());

        for (auto& packet : m_packets)
            if (!packet->complete() && packet->ready())
                packet->wait();
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

template <typename T, typename Allocator> class halo_exchange_batched {
  private:
    std::vector<halo_exchange<T, Allocator>> m_tasks;

  public:
    halo_exchange_batched(const MPI_Comm& parent_comm, const ac::shape& global_nn,
                          const ac::index& rr, const size_t nbatches)
    {
        for (size_t i{0}; i < nbatches; ++i)
            m_tasks.push_back(halo_exchange<T, Allocator>{parent_comm, global_nn, rr});
    }

    void launch(const std::vector<ac::mr::pointer<T, Allocator>>& inputs,
                std::vector<ac::mr::pointer<T, Allocator>>        outputs)
    {
        ERRCHK_MPI(inputs.size() == outputs.size());
        ERRCHK_MPI(inputs.size() <= m_tasks.size());

        for (size_t i{0}; i < inputs.size(); ++i)
            m_tasks[i].launch(inputs[i], outputs[i]);
    }

    void wait()
    {
        while (!complete())
            for (auto& task : m_tasks)
                if (!task.complete() && task.ready())
                    task.wait_one();
    }

    /** Returns true if all messages have been completed */
    bool complete() const
    {
        for (const auto& task : m_tasks)
            if (!task.complete())
                return false;

        return true;
    }
};
} // namespace ac::mpi
