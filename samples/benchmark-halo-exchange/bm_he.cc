#include <cstdlib>
#include <iostream>

#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/print_debug.h"

#include "acm/detail/cuda_utils.h"

static void
halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    ac::host_ndbuffer<double> hbuf{local_mm};
    std::iota(hbuf.begin(), hbuf.end(), 1);

    const auto din{hbuf.to_device()};
    ac::device_ndbuffer<double> dout{local_mm};

    ac::comm::async_halo_exchange_task<double> he{local_mm, local_nn, rr, 1};
    he.launch(cart_comm, {din.get()});
    he.wait({dout.get()});

    const auto start{std::chrono::system_clock::now()};
    he.launch(cart_comm, {din.get()});
    he.wait({dout.get()});
    const auto us_elapsed{std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start)};
    std::cout << "Rank " << ac::mpi::get_rank(cart_comm) << "/" << ac::mpi::get_size(cart_comm)
              << ": " << us_elapsed.count() << " us" << std::endl;
}

int
main(int argc, char** argv)
{
    ac::mpi::init_funneled();
    try {

        // Select device
        const int original_rank{ac::mpi::get_rank(MPI_COMM_WORLD)};
        const int ndevices_per_node{8};
        const int device_id{original_rank % ndevices_per_node};
        ERRCHK_CUDA_API(cudaSetDevice(device_id));

        const size_t nx = (argc > 1) ? (size_t)atol(argv[1]) : 128;
        const size_t ny = (argc > 2) ? (size_t)atol(argv[2]) : 128;
        const size_t nz = (argc > 3) ? (size_t)atol(argv[3]) : 128;

        PRINT_DEBUG(nx);
        PRINT_DEBUG(ny);
        PRINT_DEBUG(nz);

        const ac::shape global_nn{nx, ny, nz};
        const ac::index rr{3, 3, 3};

        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn,
                                                     ac::mpi::RankReorderMethod::hierarchical)};

        halo_exchange(cart_comm, global_nn, rr);

        ac::mpi::cart_comm_destroy(&cart_comm);
    }
    catch (const std::exception& e) {
        ERROR_DESC("Exception caught");
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
