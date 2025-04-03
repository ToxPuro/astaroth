#include <cstdlib>

#include "acm/detail/allocator.h"
#include "acm/detail/halo_exchange.h"
#include "acm/detail/halo_exchange_batched.h"
#include "acm/detail/halo_exchange_custom.h"
#include "acm/detail/halo_exchange_mpi.h"
#include "acm/detail/halo_exchange_packed.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/timer.h"

#include <array>

// constexpr size_t NSAMPLES{10};

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
verify_mpi_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ac::host_ndbuffer<T> gmesh{global_nn};
    std::iota(gmesh.begin(), gmesh.end(), 0);

    const auto           local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto           local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::host_ndbuffer<T> host_lmesh{local_mm, -1};
    auto                 lmesh{host_lmesh.to_device()};

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
    host_lmesh = lmesh.to_host();

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t i{0}; i < prod(local_mm); ++i) {
        const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                           global_nn};
        const auto linear_idx{to_linear(lcoords, global_nn)};
        ERRCHK(within_machine_epsilon(host_lmesh[i], static_cast<T>(linear_idx)));
    }

    // Benchmark
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // ac::timer t;
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // for (size_t i{0}; i < NSAMPLES; ++i) {
    //     reqs = launch_halo_exchange(cart_comm, local_mm, local_nn, rr, lmesh.data(),
    //     lmesh.data()); while (!reqs.empty()) {
    //         ac::mpi::request_wait_and_destroy(&reqs.back());
    //         reqs.pop_back();
    //     }
    // }
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // if (ac::mpi::get_rank(cart_comm) == 0)
    //     t.print_lap("mpi NSAMPLES TOTAL");

    return 0;
}

template <typename T>
static int
verify_custom_packed_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                                   const ac::index& rr)
{
    std::array<ac::host_ndbuffer<T>, 4> gbufs{
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
    };
    for (size_t i{0}; i < gbufs.size(); ++i)
        std::iota(gbufs[i].begin(), gbufs[i].end(), i * prod(global_nn));

    const auto                            local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                            local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    std::array<ac::device_ndbuffer<T>, 4> lbufs{
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
    };

    for (size_t i{0}; i < gbufs.size(); ++i)
        ac::mpi::scatter_advanced(cart_comm,
                                  ac::mpi::get_dtype<T>(),
                                  global_nn,
                                  ac::make_index(global_nn.size(), 0),
                                  gbufs[i].data(),
                                  local_mm,
                                  local_nn,
                                  rr,
                                  lbufs[i].data());

    std::vector<ac::mr::device_pointer<T>> lptrs;
    for (auto& lbuf : lbufs)
        lptrs.push_back(lbuf.get());

    ac::comm::async_halo_exchange_task<T> he{local_mm, local_nn, rr, lbufs.size()};
    he.launch(cart_comm, lptrs);
    he.wait(lptrs);

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t j{0}; j < lbufs.size(); ++j) {
        const auto lbuf{lbufs[j].to_host()};
        for (uint64_t i{0}; i < prod(local_mm); ++i) {
            const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                               global_nn};
            const auto linear_idx{to_linear(lcoords, global_nn)};
            ERRCHK(
                within_machine_epsilon(lbuf[i], static_cast<T>(linear_idx + j * prod(global_nn))));
        }
    }

    // Benchmark
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // ac::timer t;
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // for (size_t i{0}; i < NSAMPLES; ++i) {
    //     he.launch(cart_comm, lptrs);
    //     he.wait(lptrs);
    // }
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // if (ac::mpi::get_rank(cart_comm) == 0)
    //     t.print_lap("custom NSAMPLES TOTAL");

    return 0;
}

template <typename T>
static int
verify_revised_mpi_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                                 const ac::index& rr)
{
    std::array<ac::host_ndbuffer<T>, 4> gbufs{
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
    };
    for (size_t i{0}; i < gbufs.size(); ++i)
        std::iota(gbufs[i].begin(), gbufs[i].end(), i * prod(global_nn));

    const auto                            local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                            local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    std::array<ac::device_ndbuffer<T>, 4> lbufs{
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
    };

    for (size_t i{0}; i < gbufs.size(); ++i)
        ac::mpi::scatter_advanced(cart_comm,
                                  ac::mpi::get_dtype<T>(),
                                  global_nn,
                                  ac::make_index(global_nn.size(), 0),
                                  gbufs[i].data(),
                                  local_mm,
                                  local_nn,
                                  rr,
                                  lbufs[i].data());

    std::vector<ac::mr::device_pointer<T>> lptrs;
    for (auto& lbuf : lbufs)
        lptrs.push_back(lbuf.get());

    ac::mpi::halo_exchange_batched<T, ac::mr::device_allocator> he{cart_comm, global_nn, rr, 8};
    he.launch(lptrs, lptrs);
    he.wait();

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t j{0}; j < lbufs.size(); ++j) {
        const auto lbuf{lbufs[j].to_host()};
        for (uint64_t i{0}; i < prod(local_mm); ++i) {
            const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                               global_nn};
            const auto linear_idx{to_linear(lcoords, global_nn)};
            ERRCHK(
                within_machine_epsilon(lbuf[i], static_cast<T>(linear_idx + j * prod(global_nn))));
        }
    }

    // Benchmark
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // ac::timer t;
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // for (size_t i{0}; i < NSAMPLES; ++i) {
    //     he.launch(lptrs, lptrs);
    //     he.wait();
    // }
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // if (ac::mpi::get_rank(cart_comm) == 0)
    //     t.print_lap("revised mpi NSAMPLES TOTAL");

    return 0;
}

template <typename T>
static int
verify_custom_revised_packed_halo_exchange(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                                           const ac::index& rr)
{
    std::array<ac::host_ndbuffer<T>, 4> gbufs{
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
    };
    for (size_t i{0}; i < gbufs.size(); ++i)
        std::iota(gbufs[i].begin(), gbufs[i].end(), i * prod(global_nn));

    const auto                            local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                            local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    std::array<ac::device_ndbuffer<T>, 4> lbufs{
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
    };

    for (size_t i{0}; i < gbufs.size(); ++i)
        ac::mpi::scatter_advanced(cart_comm,
                                  ac::mpi::get_dtype<T>(),
                                  global_nn,
                                  ac::make_index(global_nn.size(), 0),
                                  gbufs[i].data(),
                                  local_mm,
                                  local_nn,
                                  rr,
                                  lbufs[i].data());

    std::vector<ac::mr::device_pointer<T>> lptrs;
    for (auto& lbuf : lbufs)
        lptrs.push_back(lbuf.get());

    acm::halo_exchange<T, ac::mr::device_allocator> he{cart_comm, global_nn, rr, lbufs.size()};
    he.launch(lptrs);
    he.wait(lptrs);

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t j{0}; j < lbufs.size(); ++j) {
        const auto lbuf{lbufs[j].to_host()};
        for (uint64_t i{0}; i < prod(local_mm); ++i) {
            const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                               global_nn};
            const auto linear_idx{to_linear(lcoords, global_nn)};
            ERRCHK(
                within_machine_epsilon(lbuf[i], static_cast<T>(linear_idx + j * prod(global_nn))));
        }
    }

    // Benchmark
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // ac::timer t;
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // for (size_t i{0}; i < NSAMPLES; ++i) {
    //     he.launch(lptrs);
    //     he.wait(lptrs);
    // }
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // if (ac::mpi::get_rank(cart_comm) == 0)
    //     t.print_lap("custom revised NSAMPLES TOTAL");

    return 0;
}

template <typename T>
static int
verify_custom_revised_packed_halo_exchange_v2(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                                              const ac::index& rr)
{
    std::array<ac::host_ndbuffer<T>, 4> gbufs{
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
        ac::host_ndbuffer<T>{global_nn},
    };
    for (size_t i{0}; i < gbufs.size(); ++i)
        std::iota(gbufs[i].begin(), gbufs[i].end(), i * prod(global_nn));

    const auto                            local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto                            local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    std::array<ac::device_ndbuffer<T>, 4> lbufs{
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
        ac::host_ndbuffer<T>{local_mm, -1}.to_device(),
    };

    for (size_t i{0}; i < gbufs.size(); ++i)
        ac::mpi::scatter_advanced(cart_comm,
                                  ac::mpi::get_dtype<T>(),
                                  global_nn,
                                  ac::make_index(global_nn.size(), 0),
                                  gbufs[i].data(),
                                  local_mm,
                                  local_nn,
                                  rr,
                                  lbufs[i].data());

    std::vector<ac::mr::device_pointer<T>> lptrs;
    for (auto& lbuf : lbufs)
        lptrs.push_back(lbuf.get());

    acm::rev::halo_exchange<T, ac::mr::device_allocator> he{cart_comm, global_nn, rr, lbufs.size()};
    he.launch(lptrs);
    he.wait(lptrs);

    const auto global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    for (uint64_t j{0}; j < lbufs.size(); ++j) {
        const auto lbuf{lbufs[j].to_host()};
        for (uint64_t i{0}; i < prod(local_mm); ++i) {
            const auto lcoords{(global_nn + global_nn_offset + ac::to_spatial(i, local_mm) - rr) %
                               global_nn};
            const auto linear_idx{to_linear(lcoords, global_nn)};
            ERRCHK(
                within_machine_epsilon(lbuf[i], static_cast<T>(linear_idx + j * prod(global_nn))));
        }
    }

    // Benchmark
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // ac::timer t;
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // for (size_t i{0}; i < NSAMPLES; ++i) {
    //     he.launch(lptrs);
    //     he.wait(lptrs);
    // }
    // ERRCHK_MPI_API(MPI_Barrier(MPI_COMM_WORLD));
    // if (ac::mpi::get_rank(cart_comm) == 0)
    //     t.print_lap("custom revised v2 NSAMPLES TOTAL");

    return 0;
}

static void
verify(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    ERRCHK_MPI(verify_mpi_halo_exchange<int>(cart_comm, global_nn, rr) == 0);
    ERRCHK_MPI(verify_revised_mpi_halo_exchange<int>(cart_comm, global_nn, rr) == 0);
    ERRCHK_MPI(verify_custom_packed_halo_exchange<int>(cart_comm, global_nn, rr) == 0);
    ERRCHK_MPI(verify_custom_revised_packed_halo_exchange<int>(cart_comm, global_nn, rr) == 0);
    ERRCHK_MPI(verify_custom_revised_packed_halo_exchange_v2<int>(cart_comm, global_nn, rr) == 0);
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        const ac::shape global_nn{ac::make_shape(3, 16)};
        const auto      rr{ac::make_index(global_nn.size(), 3)};

        {
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                         global_nn,
                                                         ac::mpi::RankReorderMethod::hierarchical)};

            verify(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                         global_nn,
                                                         ac::mpi::RankReorderMethod::no)};
            verify(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            MPI_Comm cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD,
                                                         global_nn,
                                                         ac::mpi::RankReorderMethod::default_mpi)};
            verify(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
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
