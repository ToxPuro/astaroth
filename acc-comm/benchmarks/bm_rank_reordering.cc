#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <unistd.h>

#include "acm/detail/allocator.h"
#include "acm/detail/buffer.h"
#include "acm/detail/convert.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/halo_exchange_custom.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/pack.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"

#if defined(ACM_DEVICE_ENABLED)
#include "acm/detail/cuda_utils.h"
#include "acm/detail/errchk_cuda.h"
#endif

#include "acm/detail/experimental/mpi_utils_experimental.h"
#include "acm/detail/experimental/random_experimental.h"
#include "bm.h"

namespace ac::mpi {

template <typename T, typename Allocator> class packet {
  private:
    ac::mpi::comm            m_comm;
    ac::buffer<T, Allocator> m_send_buffer;
    ac::buffer<T, Allocator> m_recv_buffer;
    ac::segment              m_recv_segment;

    ac::mpi::request m_send_req{MPI_REQUEST_NULL};
    ac::mpi::request m_recv_req{MPI_REQUEST_NULL};

    int     m_recv_neighbor{MPI_PROC_NULL};
    int     m_send_neighbor{MPI_PROC_NULL};
    int16_t m_tag{0};

  public:
    packet(const MPI_Comm& parent_comm, const ac::shape& global_nn, const ac::index& rr,
           const ac::segment& segment)
        : m_comm{parent_comm},
          m_send_buffer{prod(segment.dims)},
          m_recv_buffer{prod(segment.dims)},
          m_recv_segment{segment}
    {
        const auto          local_nn{ac::mpi::get_local_nn(parent_comm, global_nn)};
        const ac::direction recv_direction{ac::mpi::get_direction(segment.offset, local_nn, rr)};
        m_recv_neighbor = ac::mpi::get_neighbor(parent_comm, recv_direction);
        m_send_neighbor = ac::mpi::get_neighbor(parent_comm, -recv_direction);
    }

    /** Initialize the buffers for verification */
    void init()
    {
        const auto rank{ac::mpi::get_rank(m_comm.get())};
        auto       tmp{m_send_buffer.to_host()};
        std::iota(tmp.begin(), tmp.end(), as<size_t>(rank) * m_send_buffer.size());
        migrate(tmp, m_send_buffer);

        std::fill(tmp.begin(), tmp.end(), 0);
        migrate(tmp, m_recv_buffer);
    }

    void reset() {
        acm::experimental::randomize(m_send_buffer.get());
    }

    /** Verify that the contents of the recv buffer are as expected */
    void verify() const
    {
        auto tmp{m_recv_buffer.to_host()};
        for (size_t i{0}; i < m_recv_buffer.size(); ++i)
            ERRCHK_MPI(
                within_machine_epsilon(tmp[i], i + as<T>(m_recv_neighbor) * m_send_buffer.size()));
    }

    void pack(const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
              const ac::mr::pointer<T, Allocator>& input)
    {
        const ac::segment m_send_segment{m_recv_segment.dims,
                                         ((local_nn + m_recv_segment.offset - local_rr) %
                                          local_nn) +
                                             local_rr};
        acm::pack(local_mm,
                  m_send_segment.dims,
                  m_send_segment.offset,
                  {input},
                  m_send_buffer.get());
    }

    void unpack(const ac::shape& local_mm, ac::mr::pointer<T, Allocator> output)
    {
        acm::unpack(m_recv_buffer.get(),
                    local_mm,
                    m_recv_segment.dims,
                    m_recv_segment.offset,
                    {output});
    }

    void display() const
    {
        MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD)
        m_send_buffer.display();
        m_recv_buffer.display();
        MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD)
    }

    void launch()
    {
        ERRCHK_MPI(m_send_req.complete());
        ERRCHK_MPI(m_recv_req.complete());

        const auto rank{ac::mpi::get_rank(m_comm.get())};
        if (rank == m_recv_neighbor && rank == m_send_neighbor) {
            migrate(m_send_buffer, m_recv_buffer);

            // Create a dummy request to simplify asynchronous management
            static int dummy{-1};
            ERRCHK_MPI_API(
                MPI_Irecv(&dummy, 0, MPI_INT, rank, m_tag, m_comm.get(), m_recv_req.data()));
            ERRCHK_MPI_API(
                MPI_Isend(&dummy, 0, MPI_INT, rank, m_tag, m_comm.get(), m_send_req.data()));
        }
        else { // Use MPI communication
            ERRCHK_MPI_API(MPI_Irecv(m_recv_buffer.data(),
                                     as<int>(m_recv_buffer.size()),
                                     ac::mpi::get_dtype<T>(),
                                     m_recv_neighbor,
                                     m_tag,
                                     m_comm.get(),
                                     m_recv_req.data()));

            ERRCHK_MPI_API(MPI_Isend(m_send_buffer.data(),
                                     as<int>(m_send_buffer.size()),
                                     ac::mpi::get_dtype<T>(),
                                     m_send_neighbor,
                                     m_tag,
                                     m_comm.get(),
                                     m_send_req.data()));
        }

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

    void init()
    {
        for (auto& packet : m_packets)
            packet.init();
    }

    void reset() {
        for (auto& packet : m_packets)
            packet.reset();
    }

    void verify() const
    {
        for (auto& packet : m_packets)
            packet.verify();
    }

    void pack(const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
              const ac::mr::pointer<T, Allocator>& input)
    {
        for (auto& packet : m_packets)
            packet.pack(local_mm, local_nn, local_rr, input);
    }

    void unpack(const ac::shape& local_mm, ac::mr::pointer<T, Allocator> output)
    {
        for (auto& packet : m_packets)
            packet.unpack(local_mm, output);
    }

    void display() const
    {
        for (const auto& packet : m_packets)
            packet.display();
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

template <typename T, typename Allocator>
static void
set_to_global_iota(const MPI_Comm& comm, const ac::shape& global_nn, const ac::index& rr,
                   ac::mr::pointer<T, Allocator> out)
{
    const auto           local_mm{ac::mpi::get_local_mm(comm, global_nn, rr)};
    const auto           local_nn{ac::mpi::get_local_nn(comm, global_nn)};
    ac::host_ndbuffer<T> tmp{local_nn};

    for (uint64_t i{0}; i < tmp.size(); ++i) {
        const auto local_coords{ac::to_spatial(i, local_nn)};
        const auto global_coords{local_coords + ac::mpi::get_global_nn_offset(comm, global_nn)};
        const auto global_index{ac::to_linear(global_coords, global_nn)};
        tmp[i] = global_index + 1;
    }
    ac::mr::copy(tmp.get(), out);
}

/**
 * Strategy
 * 1. Set local mesh to global linear index
 * 2. Pack
 * 3. Reset local mesh to 0 (to detect overwrites during unpacking
 * 4. Exchange data
 * 5. Unpack
 * 6. Check that inner domain is zero, boundaries correspond to global linear index
 */
template <typename T, typename Allocator>
static void
verify_results(const MPI_Comm& comm, const ac::shape& global_nn, const ac::index& rr,
               const ac::ndbuffer<T, Allocator>& input)
{
    const auto ref{input.to_host()};

    const auto local_mm{ac::mpi::get_local_mm(comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(comm, global_nn)};
    const auto global_nn_offset{ac::mpi::get_global_nn_offset(comm, global_nn)};
    for (uint64_t i{0}; i < prod(local_mm); ++i) {
        const auto lcoords{ac::to_spatial(i, local_mm)};
        const auto gcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                           global_nn};

        if (ac::within_box(lcoords, local_nn, rr)) {
            ERRCHK(within_machine_epsilon(ref[i], static_cast<T>(0)));
        }
        else {
            const auto linear_idx{to_linear(gcoords, global_nn)};
            ERRCHK(within_machine_epsilon(ref[i], static_cast<T>(linear_idx + 1)));
        }
    }
}

template <typename T, typename Allocator>
static void
verify(const MPI_Comm& comm, const ac::shape& global_nn, const ac::index& rr,
       ac::mpi::halo_exchange<T, Allocator>& task, const std::function<void()>& bench)
{
    // Simple verification
    task.init();
    bench();
    task.verify();
    // task.display();

    // Exhaustive verification
    const auto                       local_mm{ac::mpi::get_local_mm(comm, global_nn, rr)};
    const auto                       local_nn{ac::mpi::get_local_nn(comm, global_nn)};
    const ac::ndbuffer<T, Allocator> tmp{local_nn};
    set_to_global_iota(comm, global_nn, rr, tmp.get());

    const ac::ndbuffer<T, Allocator> din{ac::host_ndbuffer<T>{local_mm, 0}.to_device()};
    acm::unpack(tmp.get(), local_mm, local_nn, rr, {din.get()});
    // MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    // din.display();
    // MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);
    task.pack(local_mm, local_nn, rr, din.get());

    ac::host_ndbuffer<T> tmp2{local_mm, 0};
    ac::mr::copy(tmp2.get(), din.get());

    bench();
    task.unpack(local_mm, din.get());
    // MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
    // din.display();
    // MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);
    verify_results(comm, global_nn, rr, din);
}

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        using T         = uint64_t;
        using Allocator = ac::mr::device_allocator;

#if defined(ACM_DEVICE_ENABLED)
        int device_count{0};
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        int device_id{ac::mpi::get_rank(MPI_COMM_WORLD) % device_count};
        if (device_count == 8) { // Do manual GPU mapping for LUMI
            ac::ntuple<int> device_ids{6, 7, 0, 1, 2, 3, 4, 5};
            device_id = device_ids[as<size_t>(device_id)];
        }
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
#endif

        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0)
            std::cerr << "Usage: ./bm_rank_reordering <nx> <ny> <nz> <radius> <nsamples> "
                         "<jobid>"
                      << std::endl;
        const size_t nx{(argc > 1) ? std::stoull(argv[1]) : 32};
        const size_t ny{(argc > 2) ? std::stoull(argv[2]) : 32};
        const size_t nz{(argc > 3) ? std::stoull(argv[3]) : 32};
        const size_t radius{(argc > 4) ? std::stoull(argv[4]) : 3};
        const size_t nsamples{(argc > 5) ? std::stoull(argv[5]) : 10};
        const size_t jobid{(argc > 6) ? std::stoull(argv[6]) : 0};

        const ac::shape global_nn{nx, ny, nz};
        const ac::index rr{ac::make_index(global_nn.size(), radius)};

        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0) {
            PRINT_DEBUG(nx);
            PRINT_DEBUG(ny);
            PRINT_DEBUG(nz);
            PRINT_DEBUG(radius);
            PRINT_DEBUG(nsamples);
            PRINT_DEBUG(jobid);
        }

        std::ostringstream oss;
        oss << "bm-rank-reordering-" << jobid << "-" << getpid() << "-" << ac::mpi::get_rank(MPI_COMM_WORLD) << ".csv";
        const auto output_file{oss.str()};
        FILE*      fp{fopen(output_file.c_str(), "w")};
        ERRCHK(fp);
        fprintf(fp, "impl,nx,ny,nz,radius,sample,nsamples,jobid,ns\n");
        ERRCHK(fclose(fp) == 0);

        auto print = [&](const std::string&                                label,
                         const std::vector<std::chrono::nanoseconds::rep>& results) {
            FILE* fp{fopen(output_file.c_str(), "a")};
            ERRCHK(fp);

            for (size_t i{0}; i < results.size(); ++i) {
                fprintf(fp, "%s", label.c_str());
                fprintf(fp, ",%zu", nx);
                fprintf(fp, ",%zu", ny);
                fprintf(fp, ",%zu", nz);
                fprintf(fp, ",%zu", radius);
                fprintf(fp, ",%zu", i);
                fprintf(fp, ",%zu", nsamples);
                fprintf(fp, ",%zu", jobid);
                fprintf(fp, ",%lld", as<long long>(results[i]));
                fprintf(fp, "\n");
            }
            ERRCHK(fclose(fp) == 0);
        };

        auto bm   = [&](const std::string& label, const ac::mpi::RankReorderMethod reorder_method) {
            ac::mpi::cart_comm cart_comm{MPI_COMM_WORLD, global_nn, reorder_method};
            ac::mpi::halo_exchange<T, Allocator> task{cart_comm.get(), global_nn, rr};
            auto init = [&task]() { task.reset(); };
            auto                                 bench = [&task]() {
                task.launch();
                task.wait();
            };
            auto sync = []() { ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD)); };

            // Print rank ordering
            const auto rank_ordering{ac::mpi::get_rank_ordering(cart_comm.get())};
            if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0) {
                for (const auto& coords : rank_ordering)
                    std::cout << label << ": " << coords << std::endl;
                PRINT_DEBUG(ac::mpi::get_decomposition(cart_comm.get()));
                PRINT_DEBUG(ac::mpi::get_local_nn(cart_comm.get(), global_nn));
            }

            verify(cart_comm.get(), global_nn, rr, task, bench);
            print(label, bm::benchmark(init, bench, sync, nsamples));
        };

        bm("hierarchical", ac::mpi::RankReorderMethod::hierarchical);
        bm("mpi-default", ac::mpi::RankReorderMethod::default_mpi);
        bm("mpi-no", ac::mpi::RankReorderMethod::no);
        bm("mpi-no-custom-decomp", ac::mpi::RankReorderMethod::no_custom_decomp);
        bm("mpi-default-custom-decomp", ac::mpi::RankReorderMethod::no_custom_decomp);
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
