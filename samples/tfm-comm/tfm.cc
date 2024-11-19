#include <cstdlib>
#include <iostream>

#include <numeric>

#include "datatypes.h"
#include "ndarray.h"

#include "mpi_utils.h"
#include <mpi.h>

#include "halo_exchange_packed.h"
#include "io.h"

#include "errchk.h"
#include "print_debug.h"

template <typename T, size_t N, size_t M>
static void
compute_loop(const MPI_Comm& cart_comm, HaloExchangeTask<T, N>& halo_exchange,
             ac::array<T*, M>& buffers)
{
    halo_exchange.launch(cart_comm, buffers);
    halo_exchange.wait(buffers);
}

int
main()
{
    init_mpi_funneled();
    try {
        constexpr size_t ndims = 2;
        const Shape<ndims> global_nn{8, 8};

        MPI_Comm cart_comm{cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const Shape<ndims> decomp{get_decomposition<ndims>(cart_comm)};
        const Shape<ndims> local_nn{global_nn / decomp};
        const Index<ndims> coords{get_coords<ndims>(cart_comm)};
        const Index<ndims> global_nn_offset{coords * local_nn};

        const Shape<ndims> rr{as<uint64_t>(2) * ones<uint64_t, ndims>()}; // Symmetric halo
        const Shape<ndims> local_mm{as<uint64_t>(2) * rr + local_nn};
        const int rank{get_rank(cart_comm)};

        // Initialize the mesh
        NdArray<AcReal, ndims, HostMemoryResource> lnrho(local_mm);
        NdArray<AcReal, ndims, HostMemoryResource> ux(local_mm);
        NdArray<AcReal, ndims, HostMemoryResource> uy(local_mm);
        // const auto start_index = static_cast<AcReal>(rank * prod(local_mm));
        // std::iota(lnrho.begin(), lnrho.end(), start_index);
        std::fill(lnrho.begin(), lnrho.end(), static_cast<AcReal>(rank));
        std::fill(ux.begin(), ux.end(), static_cast<AcReal>(rank) + 10);
        std::fill(uy.begin(), uy.end(), static_cast<AcReal>(rank) + 20);

        // Halo exchange
        ac::array<AcReal*, 3> buffers{lnrho.data(), ux.data(), uy.data()};
        HaloExchangeTask<AcReal, ndims> halo_exchange{local_mm, local_nn, rr, buffers.size()};
        // halo_exchange.launch(cart_comm, buffers);
        // halo_exchange.wait(buffers);
        BENCHMARK((compute_loop<AcReal, ndims, 3>(cart_comm, halo_exchange, buffers)));

        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        lnrho.display();
        ux.display();
        uy.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)

        const std::string filepath = "test.dat";
        PRINT_LOG("writing...");
        mpi_write_collective(cart_comm, global_nn, global_nn_offset, local_mm, local_nn, rr,
                             lnrho.data(), filepath);

        PRINT_LOG("reading...");
        NdArray<AcReal, ndims, HostMemoryResource> global_mesh{global_nn};
        mpi_read_collective(cart_comm, global_nn, Index<ndims>{}, global_nn, global_nn,
                            Index<ndims>{}, filepath, global_mesh.data());
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm)
        if (rank == 0) {
            global_mesh.display();
        }
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm)
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        abort_mpi();
    }

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
