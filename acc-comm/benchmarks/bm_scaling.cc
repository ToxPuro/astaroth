#include <cstdlib>

#include "acm/detail/mpi_utils.h"

static void
bm_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn)
{
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        const ac::shape global_nn{256, 256, 256};
        MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

        bm_halo_exchange(cart_comm, global_nn);

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
