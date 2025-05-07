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
#include "acm/detail/halo_exchange_mpi.h"
#include "acm/detail/halo_exchange_mpi_hindexed.h"
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
#include "acm/detail/experimental/verify_experimental.h"

#include "acm/detail/experimental/bm.h"

#include "acm/detail/halo_exchange_batched.h"
#include "acm/detail/halo_exchange_custom.h"
#include "acm/detail/halo_exchange_mpi.h"

template <typename T, typename Allocator>
void
initialize_buffers_for_verification(const ac::shape& global_nn, const ac::shape& global_nn_offset,
                                    const ac::shape& local_mm, const ac::shape& local_nn,
                                    const ac::index&                           local_rr,
                                    std::vector<ac::mr::pointer<T, Allocator>> buffers)
{
    // Initialize to global iota
    for (size_t i{0}; i < buffers.size(); ++i) {
        ac::to_global_iota(global_nn,
                           global_nn_offset,
                           local_mm,
                           local_nn,
                           local_rr,
                           buffers[i],
                           static_cast<T>(i));
    }
    // Reset halos
    const auto segments{prune(partition(local_mm, local_nn, local_rr), local_nn, local_rr)};
    for (auto& buffer : buffers)
        for (auto& segment : segments)
            ac::fill_value(local_mm, segment, buffer, -1);
}

template <typename T, typename Allocator>
void
verify_buffers(const ac::shape& global_nn, const ac::shape& global_nn_offset,
               const ac::shape& local_mm, const ac::shape& local_nn, const ac::index& local_rr,
               const std::vector<ac::mr::pointer<T, Allocator>>& buffers)
{
    for (size_t i{0}; i < buffers.size(); ++i) {
        ac::verify_global_iota(global_nn,
                               global_nn_offset,
                               local_mm,
                               local_nn,
                               local_rr,
                               buffers[i],
                               static_cast<T>(i));
    }
}

int
main(int argc, char* argv[])
{
    ac::mpi::init_funneled();
    try {
        using T         = double;
        using Allocator = ac::mr::device_allocator;

        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0)
            std::cerr << "Usage: ./bm_collective_comm <nx> <ny> <nz> <radius> <npack> <nsamples> "
                         "<jobid>"
                      << std::endl;
        const size_t nx{(argc > 1) ? std::stoull(argv[1]) : 32};
        const size_t ny{(argc > 2) ? std::stoull(argv[2]) : 32};
        const size_t nz{(argc > 3) ? std::stoull(argv[3]) : 32};
        const size_t radius{(argc > 4) ? std::stoull(argv[4]) : 3};
        const size_t npack{(argc > 5) ? std::stoull(argv[5]) : 1};
        const size_t nsamples{(argc > 6) ? std::stoull(argv[6]) : 10};
        const size_t jobid{(argc > 7) ? std::stoull(argv[7]) : 0};

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
        oss << "bm-collective-comm-" << jobid << "-" << getpid() << "-"
            << ac::mpi::get_rank(MPI_COMM_WORLD) << ".csv";
        const auto output_file{oss.str()};
        FILE*      fp{fopen(output_file.c_str(), "w")};
        ERRCHK(fp);
        fprintf(fp, "impl,nx,ny,nz,radius,npack,sample,nsamples,rank,nprocs,jobid,ns\n");
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
                fprintf(fp, ",%zu", npack);
                fprintf(fp, ",%zu", i);
                fprintf(fp, ",%zu", nsamples);
                fprintf(fp, ",%d", ac::mpi::get_rank(MPI_COMM_WORLD));
                fprintf(fp, ",%d", ac::mpi::get_size(MPI_COMM_WORLD));
                fprintf(fp, ",%zu", jobid);
                fprintf(fp, ",%lld", as<long long>(results[i]));
                fprintf(fp, "\n");
            }
            ERRCHK(fclose(fp) == 0);
        };

        ac::mpi::cart_comm comm{MPI_COMM_WORLD, global_nn};
        const auto         local_mm{ac::mpi::get_local_mm(comm.get(), global_nn, rr)};
        const auto         local_nn{ac::mpi::get_local_nn(comm.get(), global_nn)};
        const auto         global_nn_offset{ac::mpi::get_global_nn_offset(comm.get(), global_nn)};

        const int device_id{ac::mpi::select_device_lumi()};
        MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD);
        PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
        PRINT_DEBUG(ac::mpi::get_rank(comm.get()));
        PRINT_DEBUG(device_id);
        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0) {
            PRINT_DEBUG(global_nn);
            PRINT_DEBUG(ac::mpi::local_nn(comm));
        }
        PRINT_DEBUG(ac::mpi::get_global_nn_offset(comm.get(), global_nn));
        PRINT_DEBUG(ac::mpi::get_coords(comm.get()));
        MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD);

        std::vector<ac::ndbuffer<T, Allocator>> inputs;
        std::generate_n(std::back_inserter(inputs), npack, [&]() {
            return ac::ndbuffer<T, Allocator>{ac::mpi::local_mm(comm, rr)};
        });

        auto verify_init = [&]() {
            initialize_buffers_for_verification(global_nn,
                                                global_nn_offset,
                                                local_mm,
                                                local_nn,
                                                rr,
                                                ac::unwrap_get(inputs));
        };
        auto verify_complete = [&]() {
            verify_buffers(global_nn,
                           global_nn_offset,
                           local_mm,
                           local_nn,
                           rr,
                           ac::unwrap_get(inputs));
        };

        auto init = [&]() {
            for (auto& input : inputs)
                acm::experimental::randomize(input.get());
        };

        auto sync = []() {
#if defined(ACM_DEVICE_ENABLED)
            ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
            ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
        };

        {
            std::string                                       label{"mpi-he"};
            std::vector<ac::mpi::halo_exchange<T, Allocator>> hes;
            std::generate_n(std::back_inserter(hes), inputs.size(), [&]() {
                return ac::mpi::halo_exchange<T, Allocator>{comm.get(), global_nn, rr};
            });

            auto bench = [&inputs, &hes]() {
                for (size_t i{0}; i < inputs.size(); ++i)
                    hes[i].launch(inputs[i].get(), inputs[i].get());
                for (size_t i{0}; i < inputs.size(); ++i)
                    hes[i].wait_all();
            };

            verify_init();
            bench();
            verify_complete();

            // verify<T, Allocator>(comm.get(), global_nn, rr, bench);
            print(label, bm::benchmark(init, bench, sync));
        }

        {
            std::string                                    label{"mpi-he-hindexed"};
            ac::mpi::hindexed::halo_exchange<T, Allocator> he{comm.get(),
                                                              global_nn,
                                                              rr,
                                                              ac::unwrap_get(inputs),
                                                              ac::unwrap_get(inputs)};

            auto bench = [&he]() {
                he.launch();
                he.wait_all();
            };

            verify_init();
            bench();
            verify_complete();

            print(label, bm::benchmark(init, bench, sync));
        }

        {
            std::string                      label{"acm-packed-he"};
            acm::halo_exchange<T, Allocator> he{comm.get(), global_nn, rr, npack};
            auto                             bench = [&inputs, &he]() {
                he.launch(ac::unwrap_get(inputs));
                he.wait(ac::unwrap_get(inputs));
            };

            verify_init();
            bench();
            verify_complete();

            print(label, bm::benchmark(init, bench, sync));
        }

        {
            std::string                           label{"acm-batched-he"};
            acm::rev::halo_exchange<T, Allocator> he{comm.get(), global_nn, rr, npack};
            auto                                  bench = [&inputs, &he]() {
                he.launch(ac::unwrap_get(inputs));
                he.wait(ac::unwrap_get(inputs));
            };

            verify_init();
            bench();
            verify_complete();

            print(label, bm::benchmark(init, bench, sync));
        }
    }
    catch (const std::exception& e) {
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
