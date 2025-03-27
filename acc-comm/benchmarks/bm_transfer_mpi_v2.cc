#include <cstdint>
#include <cstdio>
#include <iostream>

#include "bm.h"

#include "acm/detail/errchk_cuda.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/packet.h"

constexpr size_t problem_size{128 * 1024 * 1024}; // Bytes

static void
test_halo_exchange(void)
{
    const auto global_nn{ac::make_shape(3, 128)};
    const auto rr{ac::make_index(global_nn.size(), 32)};

    MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                 global_nn,
                                                 ac::mpi::RankReorderMethod::hierarchical)};

    const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    ac::device_ndbuffer<double> din{local_mm};
    ac::device_ndbuffer<double> dout{local_mm};

    auto init_fn = [&din]() { randomize(din.get()); };

    ac::comm::async_halo_exchange_task<double> he{local_mm, local_nn, rr, 1};
    he.launchwait(cart_comm, {din.get()}, {dout.get()});

    auto bm_fn = [&]() { he.launchwait(cart_comm, {din.get()}, {dout.get()}); };

    const auto ns_elapsed{benchmark_ns("halo exchange", init_fn, bm_fn)};

    MPI_SYNCHRONOUS_BLOCK_START(MPI_COMM_WORLD)
    std::cout << ns_elapsed << std::endl;
    MPI_SYNCHRONOUS_BLOCK_END(MPI_COMM_WORLD)

    ac::mpi::cart_comm_destroy(&cart_comm);
    ac::mpi::finalize();
    exit(EXIT_SUCCESS);
}

int
main(void)
{
    ac::mpi::init_funneled();
    try {
        // Communicator information
        const int rank{ac::mpi::get_rank(MPI_COMM_WORLD)};
        const int nprocs{ac::mpi::get_size(MPI_COMM_WORLD)};

// Set device id
#if 0
            int ndevices_per_node{-1};
            ERRCHK_CUDA_API(cudaGetDeviceCount(&ndevices_per_node));
            const int device_id{rank % ndevices_per_node};
            ERRCHK_CUDA_API(cudaSetDevice(device_id));
#else
        // Run with
        // `srun --cpu-bind=map_cpu:33,41,49,57,17,25,1,9 --account=<project> -t 00:05:00 -p
        // dev-g --gpus-per-node=8 --ntasks-per-node=8 --nodes=1 benchmarks/bm_transfer_mpi_v2`
        // to get a mapping where close-by GPUs have subsequent ranks
        ac::ntuple<int> device_ids{6, 7, 0, 1, 2, 3, 4, 5};
        const int       device_id{device_ids[as<size_t>(rank)]};
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
#endif

        test_halo_exchange();

        ac::device_buffer<uint8_t> din{problem_size};
        ac::device_buffer<uint8_t> dout{problem_size};

        if (rank == 0)
            std::cout << "Benchmarking. Note: the numbers are only for reference: may not be "
                         "properly calculated and synced, but should give a rough idea on which "
                         "path is the fastest and their comparable throughputs."
                      << std::endl;

        for (int root{0}; root < nprocs; root += 2) {
            ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
            if (rank == root)
                std::cout << "Root " << rank << std::endl;
            for (int i{0}; i < nprocs; ++i) {

                auto bm_fn = [&root, &rank, &din, &dout, &i]() {
                    MPI_Request send_req{MPI_REQUEST_NULL};
                    MPI_Request recv_req{MPI_REQUEST_NULL};

                    if (rank == root) {
                        ERRCHK_MPI_API(MPI_Irecv(dout.data(),
                                                 as<int>(dout.size()),
                                                 MPI_BYTE,
                                                 i,
                                                 i,
                                                 MPI_COMM_WORLD,
                                                 &recv_req));
                    }

                    if (rank == i) {
                        ERRCHK_MPI_API(MPI_Isend(din.data(),
                                                 as<int>(din.size()),
                                                 MPI_BYTE,
                                                 root,
                                                 i,
                                                 MPI_COMM_WORLD,
                                                 &send_req));
                        ERRCHK_MPI_API(MPI_Wait(&send_req, MPI_STATUS_IGNORE));
                    }

                    if (rank == root) {
                        ERRCHK_MPI_API(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));
                    }
                };

                constexpr size_t len{128};
                char             label[len];
                snprintf(label, len, "memory transfer (to rank %d, device %d)", i, device_id);

                const auto ns_elapsed{benchmark_ns(label, []() {}, bm_fn)};
                const auto bw{(2 * problem_size) / (1e-9 * ns_elapsed) / (1024ul * 1024 * 1024)};
                if (rank == root)
                    std::cout << "\t" << label << " median bandwidth: " << bw << " GiB/s"
                              << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        ERROR_DESC("Exception caught");
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
