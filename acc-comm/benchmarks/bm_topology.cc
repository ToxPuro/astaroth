#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "acm/detail/buffer.h"
#include "acm/detail/errchk_cuda.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/experimental/bm.h"

constexpr size_t problem_size{128 * 1024 * 1024}; // Bytes
constexpr size_t bench_nsamples{100};

int
main(void)
{
    ac::mpi::init_funneled();
    try {
        // Communicator information
        const int rank{ac::mpi::get_rank(MPI_COMM_WORLD)};
        const int nprocs{ac::mpi::get_size(MPI_COMM_WORLD)};

        int device_id{-1};
        int ndevices_per_node{-1};
#if defined(ACM_DEVICE_ENABLED)
        // Run with
        // `srun --cpu-bind=map_cpu:33,41,49,57,17,25,1,9 --account=<project> -t 00:05:00 -p
        // dev-g --gpus-per-node=8 --ntasks-per-node=8 --nodes=1 benchmarks/bm_transfer_mpi_v2`
        // to get a mapping where close-by GPUs have subsequent ranks
        ERRCHK_CUDA_API(cudaGetDeviceCount(&ndevices_per_node));
        device_id = rank % ndevices_per_node;
        ac::ntuple<int> device_ids{6, 7, 0, 1, 2, 3, 4, 5};
        if (ndevices_per_node >= 8) {
            device_id = device_ids[as<size_t>(rank)];
        }
        ERRCHK_CUDA_API(cudaSetDevice(device_id));
#endif

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
                std::cout << "Root " << rank << ". Using device " << device_id << "." << std::endl;
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
#if defined(ACM_DEVICE_ENABLED)
		if (ndevices_per_node >= 8)
                	snprintf(label, len, "memory transfer (to rank %d, device %d)", i, device_ids[i]);
		else
#endif
                	snprintf(label, len, "memory transfer (to rank %d, device %d)", i, i % ndevices_per_node);

                const auto results{bm::benchmark([]() {}, bm_fn, []() {}, bench_nsamples)};
                const auto median_ns{bm::median<std::chrono::nanoseconds>(results)};

                const auto bw{(2 * problem_size) / (1e-9 * median_ns) / (1024ul * 1024 * 1024)};
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
