#include <cstdlib>

#include "bm.h"

#include "acm/detail/halo_exchange.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"

/** Benchmark async MPI halo exhange in a Cartesian grid*/
static void
bm_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    const auto                  local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                  local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::device_ndbuffer<double> din{local_mm};
    ac::device_ndbuffer<double> dout{local_mm};

    auto init_fn = [&din]() { randomize(din.get()); };
    auto bm_fn   = [&]() {
        auto reqs{launch_halo_exchange(cart_comm, local_mm, local_nn, rr, din.data(), dout.data())};

        while (!reqs.empty()) {
            ac::mpi::request_wait_and_destroy(&reqs.back());
            reqs.pop_back();
        }
    };

    const auto median{benchmark("halo exchange", init_fn, bm_fn)};
    if (ac::mpi::get_rank(cart_comm) == 0) {
        FILE* fp{fopen("scaling.csv", "a")};
        ERRCHK(fp != NULL);

        const auto nprocs{ac::mpi::get_size(cart_comm)};
        ERRCHK(fprintf(fp, "%d,%g\n", nprocs, median) > 0);

        fclose(fp);
    }
}

/** Check that packed halo exchange gives similar performance with one array */
// static void
// bm_sanity_check(const MPI_Comm& cart_comm, const ac::shape& global_nn)
// {
// }

int
main()
{
    ac::mpi::init_funneled();
    try {
        const ac::shape global_nn{128, 128, 128};
        const ac::index rr{3, 3, 3};

        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                     global_nn,
                                                     ac::mpi::RankReorderMethod::hierarchical)};
        bm_halo_exchange(cart_comm, global_nn, rr);
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
