#include <cstdlib>
#include <iostream>

#include <numeric>

#include "datatypes.h"
#include "ndbuffer.h"

#include "mpi_utils.h"
#include <mpi.h>

#include "halo_exchange_packed.h"
#include "io.h"

#include "errchk.h"
#include "print_debug.h"

constexpr size_t ndims = 2;
using AcReal           = double;
using Shape            = ac::shape<ndims>;
using Index            = ac::index<ndims>;
using Vector           = ac::buffer<AcReal, HostMemoryResource>;
using NdVector         = ac::ndbuffer<AcReal, ndims, HostMemoryResource>;
using HaloExchange     = HaloExchangeTask<AcReal, ndims, HostMemoryResource>;

static void
compute_loop(const MPI_Comm& cart_comm, HaloExchange& halo_exchange, std::vector<Vector*>& buffers)
{
    halo_exchange.launch(cart_comm, buffers);
    halo_exchange.wait(buffers);
}

int
main()
{
    init_mpi_funneled();
    try {

        const Shape global_nn{8, 8};
        MPI_Comm cart_comm{cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const Shape decomp{get_decomposition<ndims>(cart_comm)};
        const Shape local_nn{global_nn / decomp};
        const Index coords{get_coords<ndims>(cart_comm)};
        const Index global_nn_offset{coords * local_nn};

        const Shape rr{as<uint64_t>(2) * ones<uint64_t, ndims>()}; // Symmetric halo
        const Shape local_mm{as<uint64_t>(2) * rr + local_nn};
        const int rank{get_rank(cart_comm)};

        // Initialize the mesh
        NdVector lnrho(local_mm);
        NdVector ux(local_mm);
        NdVector uy(local_mm);
        // const auto start_index = static_cast<AcReal>(rank * prod(local_mm));
        // std::iota(lnrho.begin(), lnrho.end(), start_index);
        std::fill(lnrho.begin(), lnrho.end(), static_cast<AcReal>(rank));
        std::fill(ux.begin(), ux.end(), static_cast<AcReal>(rank) + 10);
        std::fill(uy.begin(), uy.end(), static_cast<AcReal>(rank) + 20);

        // Halo exchange
        std::vector<Vector*> buffers{&lnrhobuffer, &uxbuffer, &uybuffer};
        HaloExchange halo_exchange{local_mm, local_nn, rr, buffers.size()};
        // halo_exchange.launch(cart_comm, buffers);
        // halo_exchange.wait(buffers);
        BENCHMARK((compute_loop(cart_comm, halo_exchange, buffers)));

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
        NdVector global_mesh{global_nn};
        mpi_read_collective(cart_comm, global_nn, Index{}, global_nn, global_nn, Index{}, filepath,
                            global_mesh.data());
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
