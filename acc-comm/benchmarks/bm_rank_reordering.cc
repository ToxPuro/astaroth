#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>

#include "acm/detail/allocator.h"
#include "acm/detail/convert.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/halo_exchange_custom.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"

#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

#include "bm.h"
#include "mpi_utils_experimental.h"

namespace ac::mpi {

template <typename T, typename Allocator> class packet {
  private:
    ac::mpi::comm            m_comm;
    ac::buffer<T, Allocator> m_send_buffer;
    ac::buffer<T, Allocator> m_recv_buffer;

    ac::mpi::request m_send_req{MPI_REQUEST_NULL};
    ac::mpi::request m_recv_req{MPI_REQUEST_NULL};

    int     m_recv_neighbor{MPI_PROC_NULL};
    int     m_send_neighbor{MPI_PROC_NULL};
    int16_t m_tag{0};

  public:
    packet(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
           const ac::segment& segment)
        : m_comm{parent_comm}, m_send_buffer{prod(segment.dims)}, m_recv_buffer{prod(segment.dims)}
    {
        const auto          local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
        const ac::direction recv_direction{ac::mpi::get_direction(segment.offset, local_nn, rr)};
        m_recv_neighbor = ac::mpi::get_neighbor(parent_comm, recv_direction);
        m_send_neighbor = ac::mpi::get_neighbor(parent_comm, -recv_direction);
    }

    void launch()
    {
        ERRCHK_MPI(m_send_req.complete());
        ERRCHK_MPI(m_recv_req.complete());

        MPI_Request recv_req{MPI_REQUEST_NULL};
        ERRCHK_MPI_API(MPI_Irecv(m_recv_buffer.data(),
                                 as<int>(m_recv_buffer.size()),
                                 ac::mpi::get_dtype<T>(),
                                 m_recv_neighbor,
                                 m_tag,
                                 m_comm.get(),
                                 &recv_req));
        m_recv_req = ac::mpi::request{recv_req};

        MPI_Request send_req{MPI_REQUEST_NULL};
        ERRCHK_MPI_API(MPI_Isend(m_send_buffer.data(),
                                 as<int>(m_send_buffer.size()),
                                 ac::mpi::get_dtype<T>(),
                                 m_send_neighbor,
                                 m_tag,
                                 m_comm.get(),
                                 &send_req));
        m_send_req = ac::mpi::request{send_req};

        ac::mpi::increment_tag(m_tag);
    }

    void wait()
    {
        ERRCHK_MPI(!m_send_req.complete());
        ERRCHK_MPI(!m_recv_req.complete());

        m_send_req.wait();
        m_recv_req.wait();

        ERRCHK_MPI(m_send_req.complete());
        ERRCHK_MPI(m_recv_req.complete());
    }
};

template <typename T, typename Allocator> class halo_exchange {
  private:
    std::vector<packet<T, Allocator>> m_packets;

  public:
    halo_exchange(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr)
    {
        const auto local_mm{ac::mpi::get_local_mm(parent_comm, global_nn, rr)};
        const auto local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
        const auto segments{prune(partition(local_mm, local_nn, rr), local_nn, rr)};

        for (const auto& segment : segments)
            m_packets.push_back({parent_comm, global_nn, rr, segment});
    }

    void launch()
    {
        for (auto& packet : m_packets)
            packet.launch();
    }

    void wait()
    {
        for (auto& packet : m_packets)
            packet.wait();
    }
};

} // namespace ac::mpi

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        using T         = double;
        using Allocator = ac::mr::device_allocator;

        const ac::shape    global_nn{128, 64, 8};
        const auto         rr{ac::make_index(global_nn.size(), 3)};
        ac::mpi::cart_comm cart_comm{MPI_COMM_WORLD, global_nn, ac::mpi::RankReorderMethod::no};

        ac::mpi::halo_exchange<T, Allocator> task{cart_comm.get(), global_nn, rr};

        auto init  = []() {};
        auto bench = [&task]() {
            task.launch();
            task.wait();
        };
        auto sync = []() { ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD)); };

        constexpr size_t nsamples{10};
        const auto       results{bm::benchmark(init, bench, sync, nsamples)};

        for (const auto& result : results)
            std::cout << result << std::endl;
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
