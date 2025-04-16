#include <cstdlib>
#include <iostream>
#include <sstream>

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

/** Verify halo exchange
 * Strategy:
 *     1) Set global mesh to iota
 *     2) Scatter
 *     3) Halo exchange
 *     4) Loop over the local mesh incl. halos, confirm that value at index
 *        corresponds to the global linear index
 *
 */
template <typename T, typename Allocator>
static int
verify(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ac::host_ndbuffer<T> gmesh{global_nn};
    std::iota(gmesh.begin(), gmesh.end(), 0);

    const auto                 local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                 local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::ndbuffer<T, Allocator> lmesh{local_mm};

    ac::mpi::scatter_advanced(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              ac::make_index(global_nn.size(), 0),
                              gmesh.data(),
                              local_mm,
                              local_nn,
                              rr,
                              lmesh.data());

    acm::halo_exchange<T, Allocator> task{cart_comm, global_nn, rr, 1};
    task.launch({lmesh.get()});
    task.wait({lmesh.get()});
    const auto host_lmesh{lmesh.to_host()};

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t i{0}; i < prod(local_mm); ++i) {
        const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                           global_nn};
        const auto linear_idx{to_linear(lcoords, global_nn)};
        ERRCHK(within_machine_epsilon(host_lmesh[i], static_cast<T>(linear_idx)));
    }

    return 0;
}

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0)
            std::cerr << "Usage: ./bm_rank_reordering <nx> <ny> <nz> <radius> <ninputs> <nsamples> "
                         "<jobid>"
                      << std::endl;
        const size_t nx{(argc > 1) ? std::stoull(argv[1]) : 32};
        const size_t ny{(argc > 2) ? std::stoull(argv[2]) : 32};
        const size_t nz{(argc > 3) ? std::stoull(argv[3]) : 32};
        const size_t radius{(argc > 4) ? std::stoull(argv[4]) : 3};
        const size_t ninputs{(argc > 5) ? std::stoull(argv[5]) : 1};
        const size_t nsamples{(argc > 6) ? std::stoull(argv[6]) : 10};
        const size_t jobid{(argc > 7) ? std::stoull(argv[7]) : 0};

        const ac::shape global_nn{nx, ny, nz};
        const ac::index rr{ac::make_index(global_nn.size(), radius)};

        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0) {
            PRINT_DEBUG(nx);
            PRINT_DEBUG(ny);
            PRINT_DEBUG(nz);
            PRINT_DEBUG(radius);
            PRINT_DEBUG(ninputs);
            PRINT_DEBUG(nsamples);
            PRINT_DEBUG(jobid);
        }

        std::ostringstream oss;
        oss << "bm-rank-reordering-" << jobid << "-" << ac::mpi::get_rank(MPI_COMM_WORLD) << ".csv";
        const auto output_file{oss.str()};
        FILE*      fp{fopen(output_file.c_str(), "w")};
        ERRCHK(fp);
        fprintf(fp, "impl,nx,ny,nz,radius,ninputs,sample,nsamples,jobid,ns\n");
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
                fprintf(fp, ",%zu", ninputs);
                fprintf(fp, ",%zu", i);
                fprintf(fp, ",%zu", nsamples);
                fprintf(fp, ",%zu", jobid);
                fprintf(fp, ",%lld", as<long long>(results[i]));
                fprintf(fp, "\n");
            }
            ERRCHK(fclose(fp) == 0);
        };

        // Setup the benchmark
        using T         = double;
        using Allocator = ac::mr::device_allocator;

        {
            const std::string  label{"rank-reorder-no"};
            ac::mpi::cart_comm cart_comm{MPI_COMM_WORLD, global_nn, ac::mpi::RankReorderMethod::no};
            ERRCHK_MPI((verify<T, Allocator>(cart_comm.get(), global_nn, rr) == 0));

            // Task
            const auto local_mm{ac::mpi::get_local_mm(cart_comm.get(), global_nn, rr)};
            acm::halo_exchange<T, ac::mr::device_allocator> task{cart_comm.get(),
                                                                 cart_comm.global_nn(),
                                                                 rr,
                                                                 ninputs};

            // Memory
            std::vector<ac::ndbuffer<T, Allocator>> input_buffers;
            for (size_t i{0}; i < ninputs; ++i)
                input_buffers.push_back(ac::ndbuffer<T, Allocator>{local_mm});
            auto inputs{ac::unwrap_get(input_buffers)};

            // Functions
            auto init_random = [&input_buffers]() {
                for (auto& input : input_buffers)
                    bm::randomize(input.get());
            };
            auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
                ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
                ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
            };
            auto bench = [&inputs, &task]() {
                task.launch(inputs);
                task.wait(inputs);
            };
            print(label, bm::benchmark(init_random, bench, sync, nsamples));
        }
        {
            const std::string  label{"rank-reorder-yes"};
            ac::mpi::cart_comm cart_comm{MPI_COMM_WORLD,
                                         global_nn,
                                         ac::mpi::RankReorderMethod::default_mpi};
            ERRCHK_MPI((verify<T, Allocator>(cart_comm.get(), global_nn, rr) == 0));

            // Task
            const auto local_mm{ac::mpi::get_local_mm(cart_comm.get(), global_nn, rr)};
            acm::halo_exchange<T, ac::mr::device_allocator> task{cart_comm.get(),
                                                                 cart_comm.global_nn(),
                                                                 rr,
                                                                 ninputs};

            // Memory
            std::vector<ac::ndbuffer<T, Allocator>> input_buffers;
            for (size_t i{0}; i < ninputs; ++i)
                input_buffers.push_back(ac::ndbuffer<T, Allocator>{local_mm});
            auto inputs{ac::unwrap_get(input_buffers)};

            // Functions
            auto init_random = [&input_buffers]() {
                for (auto& input : input_buffers)
                    bm::randomize(input.get());
            };
            auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
                ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
                ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
            };
            auto bench = [&inputs, &task]() {
                task.launch(inputs);
                task.wait(inputs);
            };
            print(label, bm::benchmark(init_random, bench, sync, nsamples));
        }
        {
            const std::string  label{"rank-reorder-hierarchical"};
            ac::mpi::cart_comm cart_comm{MPI_COMM_WORLD,
                                         global_nn,
                                         ac::mpi::RankReorderMethod::hierarchical};
            ERRCHK_MPI((verify<T, Allocator>(cart_comm.get(), global_nn, rr) == 0));

            // Task
            const auto local_mm{ac::mpi::get_local_mm(cart_comm.get(), global_nn, rr)};
            acm::halo_exchange<T, ac::mr::device_allocator> task{cart_comm.get(),
                                                                 cart_comm.global_nn(),
                                                                 rr,
                                                                 ninputs};

            // Memory
            std::vector<ac::ndbuffer<T, Allocator>> input_buffers;
            for (size_t i{0}; i < ninputs; ++i)
                input_buffers.push_back(ac::ndbuffer<T, Allocator>{local_mm});
            auto inputs{ac::unwrap_get(input_buffers)};

            // Functions
            auto init_random = [&input_buffers]() {
                for (auto& input : input_buffers)
                    bm::randomize(input.get());
            };
            auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
                ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
                ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
            };
            auto bench = [&inputs, &task]() {
                task.launch(inputs);
                task.wait(inputs);
            };
            print(label, bm::benchmark(init_random, bench, sync, nsamples));
        }
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
