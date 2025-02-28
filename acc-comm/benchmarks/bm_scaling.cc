#include <cstdlib>

#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/ntuple.h"
#include "bm.h"

#include "acm/detail/halo_exchange.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"

/** Verify halo exchange
 * Strategy:
 *     1) Set global mesh to iota
 *     2) Scatter
 *     3) Halo exchange
 *     4) Loop over the local mesh incl. halos, confirm that value at index
 *        corresponds to the global linear index
 *
 */
template <typename T>
static int
verify(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ac::host_ndbuffer<T> gmesh{global_nn};
    std::iota(gmesh.begin(), gmesh.end(), 0);

    const auto           local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto           local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::host_ndbuffer<T> lmesh{local_mm, -1};

    ac::mpi::scatter_advanced(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              ac::make_index(global_nn.size(), 0),
                              gmesh.data(),
                              local_mm,
                              local_nn,
                              rr,
                              lmesh.data());

    auto reqs{launch_halo_exchange(cart_comm, local_mm, local_nn, rr, lmesh.data(), lmesh.data())};
    while (!reqs.empty()) {
        ac::mpi::request_wait_and_destroy(&reqs.back());
        reqs.pop_back();
    }
    // ac::comm::async_halo_exchange_task<T> he{local_mm, local_nn, rr, 1};
    // he.launch(cart_comm, {lmesh.get()});
    // he.wait({lmesh.get()});

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t i{0}; i < prod(local_mm); ++i) {
        const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                           global_nn};
        const auto linear_idx{to_linear(lcoords, global_nn)};
        ERRCHK(within_machine_epsilon(lmesh[i], static_cast<T>(linear_idx)));
    }

    return 0;
}

/*
template <typename T>
static int
verify_packed(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ac::host_ndbuffer<T> gmesh0{global_nn};
    std::iota(gmesh0.begin(), gmesh0.end(), 0);

    const auto           local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto           local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::host_ndbuffer<T> lmesh{local_mm, -1};

    ac::mpi::scatter_advanced(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              ac::make_index(global_nn.size(), 0),
                              gmesh0.data(),
                              local_mm,
                              local_nn,
                              rr,
                              lmesh.data());

    auto reqs{launch_halo_exchange(cart_comm, local_mm, local_nn, rr, lmesh.data(), lmesh.data())};
    while (!reqs.empty()) {
        ac::mpi::request_wait_and_destroy(&reqs.back());
        reqs.pop_back();
    }
    // ac::comm::async_halo_exchange_task<T> he{local_mm, local_nn, rr, 1};
    // he.launch(cart_comm, {lmesh.get()});
    // he.wait({lmesh.get()});

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t i{0}; i < prod(local_mm); ++i) {
        const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                           global_nn};
        const auto linear_idx{to_linear(lcoords, global_nn)};
        ERRCHK(within_machine_epsilon(lmesh[i], static_cast<T>(linear_idx)));
    }

    return 0;
}
*/

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

int
main()
{
    ac::mpi::init_funneled();
    try {
        const ac::shape global_nn{8, 8, 8};
        const ac::index rr{3, 3, 3};

        MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                     global_nn,
                                                     ac::mpi::RankReorderMethod::hierarchical)};

        verify<int>(cart_comm, global_nn, rr);
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
