#include <cstdlib>
#include <limits>
#include <numeric> // std::iota

#include "acm/detail/algorithm.h"
#include "acm/detail/allocator.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/errchk_print.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/partition.h"
#include "acm/detail/print_debug.h"
#include "acm/detail/type_conversion.h"

constexpr bool verbose{true};

template <typename Allocator>
void
test_reduce_axis(const MPI_Comm& cart_comm, const ac::shape& global_nn)
{
    ac::shape    decomp{ac::mpi::get_decomposition(cart_comm)};
    ac::index    coords{ac::mpi::get_coords(cart_comm)};
    const size_t nprocs{prod(decomp)};

    // Checks that the reduce sum is the sum of all processes along a specific axis
    for (size_t axis{0}; axis < global_nn.size(); ++axis) {
        constexpr size_t                        count{10};
        const int                               value{as<int>((coords[axis] + 1) * nprocs)};
        ac::buffer<int, ac::mr::host_allocator> tmp{count, value};
        ac::buffer<int, Allocator>              buf{count};
        migrate(tmp, buf);

        BENCHMARK(ac::mpi::reduce_axis(cart_comm,
                                       ac::mpi::get_dtype<int>(),
                                       MPI_SUM,
                                       axis,
                                       buf.size(),
                                       buf.data()));

        migrate(buf, tmp);

        if (verbose) {
            PRINT_DEBUG(decomp);
            PRINT_DEBUG(coords);
            PRINT_DEBUG(nprocs);
            PRINT_DEBUG_ARRAY(tmp.size(), tmp.data());
        }

        // E.g. 4 procs on axis, the value in the buffer of each proc corresponds to its
        // coordinates on that axis
        for (size_t i{0}; i < count; ++i) {
            const size_t nprocs_on_axis{nprocs / decomp[axis]};
            ERRCHK(tmp[i] == value * as<int>(nprocs_on_axis));
        }
    }
}

void
test_scatter_gather(const MPI_Comm& cart_comm, const ac::shape& global_nn)
{
    using T      = int;
    using Buffer = ac::ndbuffer<T, ac::mr::host_allocator>;

    const ac::index global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    // const ac::index zero_offset{ac::make_index(global_nn.size(), 0)};
    const ac::shape local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    const ac::index rr{ac::make_index(global_nn.size(), 2)};
    const ac::shape local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};

    Buffer monolithic{global_nn};
    std::iota(monolithic.begin(), monolithic.end(), 1);
    Buffer distributed{local_mm};

    BENCHMARK(ac::mpi::scatter(cart_comm,
                               ac::mpi::get_dtype<T>(),
                               global_nn,
                               rr,
                               monolithic.data(),
                               distributed.data()));

    if (verbose) {
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm);
        PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
        PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
        monolithic.display();
        distributed.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm);
    }

    // Check
    Buffer monolithic_test{global_nn, 0};
    BENCHMARK(ac::mpi::gather(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              rr,
                              distributed.data(),
                              monolithic_test.data()));

    const auto rank{ac::mpi::get_rank(cart_comm)};
    if (rank == 0) {
        if (verbose) {
            PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
            PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
            monolithic.display();
            monolithic_test.display();
        }

        for (size_t i{0}; i < monolithic_test.size(); ++i)
            ERRCHK_MPI(monolithic.get()[i] == monolithic_test.get()[i]);
    }
}

void
test_scatter_gather_advanced(const MPI_Comm& cart_comm, const ac::shape& global_nn)
{
    using T      = int;
    using Buffer = ac::ndbuffer<T, ac::mr::host_allocator>;

    const ac::index global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    // const ac::index zero_offset(global_nn.size(), static_cast<int>(0));
    const ac::shape local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    const ac::index rr{ac::make_index(global_nn.size(), 2)};
    const ac::shape local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};

    const ac::shape global_mm{global_nn + static_cast<uint64_t>(2) * rr};

    Buffer monolithic{global_mm};
    std::iota(monolithic.begin(), monolithic.end(), 1);
    Buffer distributed{local_mm};

    // Scatter
    BENCHMARK(ac::mpi::scatter_advanced(cart_comm,
                                        ac::mpi::get_dtype<T>(),
                                        global_mm,
                                        rr,
                                        monolithic.data(),
                                        local_mm,
                                        local_nn,
                                        rr,
                                        distributed.data()));

    if (verbose) {
        MPI_SYNCHRONOUS_BLOCK_START(cart_comm);
        PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
        PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
        monolithic.display();
        distributed.display();
        MPI_SYNCHRONOUS_BLOCK_END(cart_comm);
    }

    // Gather
    Buffer monolithic_test{global_mm, 0}; // Initialize to zero
    BENCHMARK(ac::mpi::gather_advanced(cart_comm,
                                       ac::mpi::get_dtype<T>(),
                                       local_mm,
                                       local_nn,
                                       rr,
                                       distributed.data(),
                                       global_mm,
                                       rr,
                                       monolithic_test.data()));

    // Set boundaries to zero in the model solution
    auto segments{partition(global_mm, global_nn, rr)};
    auto it{std::remove_if(segments.begin(),
                           segments.end(),
                           [global_nn, rr](const ac::segment& segment) {
                               return within_box(segment.offset, global_nn, rr);
                           })};
    segments.erase(it, segments.end());
    for (const auto& segment : segments)
        fill(0, segment.dims, segment.offset, monolithic);

    const auto rank{ac::mpi::get_rank(cart_comm)};
    if (rank == 0) {
        if (verbose) {
            PRINT_DEBUG(ac::mpi::get_rank(MPI_COMM_WORLD));
            PRINT_DEBUG(ac::mpi::get_coords(cart_comm));
            monolithic.display();
            monolithic_test.display();
        }

        for (size_t i{0}; i < monolithic_test.size(); ++i)
            ERRCHK_MPI(monolithic.get()[i] == monolithic_test.get()[i]);
    }
}

static void
test_mpi_pack(const MPI_Comm& cart_comm, const ac::shape& global_nn)
{
    using T = int;

    const auto rr{ac::make_index(global_nn.size(), 1)};
    const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

    ac::host_ndbuffer<T> href{local_mm};
    std::iota(href.begin(), href.end(), 1);

    auto                   dref{href.to_device()};
    ac::device_ndbuffer<T> dpack{local_nn};
    ac::mpi::pack(cart_comm,
                  ac::mpi::get_dtype<T>(),
                  local_mm,
                  local_nn,
                  rr,
                  dref.data(),
                  dpack.size(),
                  dpack.data());

    ac::device_ndbuffer<T> dtst{local_mm};
    ac::mpi::unpack(cart_comm,
                    ac::mpi::get_dtype<T>(),
                    dpack.size(),
                    dpack.data(),
                    local_mm,
                    local_nn,
                    rr,
                    dtst.data());

    auto hpack{dpack.to_host()};
    auto htst{dtst.to_host()};
    href.display();
    hpack.display();
    htst.display();

    // Check that the averages match.
    // Not sure if this holds for all inputs: if this fails, it's possible
    // that packing/unpacking are still correct and only this error check is wrong.
    // Must reconsider how to check for errors if this happens.
    const auto href_avg{as<uint64_t>(std::reduce(href.begin(), href.end())) / prod(local_mm)};
    const auto htst_avg{as<uint64_t>(std::reduce(htst.begin(), htst.end())) / prod(local_nn)};
    PRINT_LOG_INFO("Checking averages of packed and unpacked buffers. Note that the error check "
                   "may itself be correct, and need to reconsider the approach if the next check "
                   "fails even with correct pack/unpack functions.");
    ERRCHK(href_avg == htst_avg);
    PRINT_LOG_INFO("OK");
}

static void
test_collective_io(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    const auto             global_mm{ac::mpi::get_global_mm(global_nn, rr)};
    const auto             global_nn_offset{ac::mpi::get_global_nn_offset(cart_comm, global_nn)};
    const auto             local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto             local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    ac::host_ndbuffer<int> ref{local_mm};

    std::iota(ref.begin(), ref.end(), ac::mpi::get_rank(cart_comm) * as<int>(ref.size()));

    ac::mpi::write_collective(cart_comm,
                              ac::mpi::get_dtype<int>(),
                              global_mm,
                              global_nn_offset,
                              local_mm,
                              local_nn,
                              rr,
                              ref.data(),
                              std::string("tmp-collective-io-test.debug"));

    ac::host_ndbuffer<int> tst{global_mm};
    ac::mpi::read_collective(cart_comm,
                             ac::mpi::get_dtype<int>(),
                             global_nn,
                             ac::make_index(global_nn.size(), 0),
                             global_mm,
                             global_nn,
                             rr,
                             std::string("tmp-collective-io-test.debug"),
                             tst.data());

    MPI_SYNCHRONOUS_BLOCK_START(cart_comm);
    ref.display();
    tst.display();
    MPI_SYNCHRONOUS_BLOCK_END(cart_comm);
}

static void
test_pack_transform_reduce(const MPI_Comm& cart_comm, const ac::shape& global_nn,
                           const ac::index& rr)
{
    using T = uint64_t;
    // const auto global_mm{ac::mpi::get_global_mm(global_nn, rr)};
    const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    const auto local_nn_offset{rr};

    ac::host_ndbuffer<T> href{global_nn};
    std::iota(href.begin(), href.end(), 1);

    ac::host_ndbuffer<T> distr_href{local_mm};
    ac::mpi::scatter_advanced(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              ac::make_index(global_nn.size(), 0),
                              href.data(),
                              local_mm,
                              local_nn,
                              local_nn_offset,
                              distr_href.data());

    MPI_SYNCHRONOUS_BLOCK_START(cart_comm);
    distr_href.display();
    MPI_SYNCHRONOUS_BLOCK_END(cart_comm);
}

#include "acm/detail/halo_exchange_packed.h"
static void
test_pipeline(const MPI_Comm& cart_comm, const ac::shape& global_nn, const ac::index& rr)
{
    using T = double;

    const auto local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
    const auto local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};
    const auto local_nn_offset{rr};

    ac::host_ndbuffer<T> href{global_nn};
    std::iota(href.begin(), href.end(), 1);

    ac::device_ndbuffer<T> distr_dref{local_mm};
    ac::mpi::scatter_advanced(cart_comm,
                              ac::mpi::get_dtype<T>(),
                              global_nn,
                              ac::make_index(global_nn.size(), 0),
                              href.data(),
                              local_mm,
                              local_nn,
                              local_nn_offset,
                              distr_dref.data());

    std::fill(href.begin(), href.end(), -1);
    ac::mpi::gather_advanced(cart_comm,
                             ac::mpi::get_dtype<T>(),
                             local_mm,
                             local_nn,
                             local_nn_offset,
                             distr_dref.data(),
                             global_nn,
                             ac::make_index(global_nn.size(), 0),
                             href.data());

    const auto rank{ac::mpi::get_rank(cart_comm)};
    if (rank == 0) {
        href.display();

        for (size_t i{0}; i < href.size(); ++i)
            ERRCHK(within_machine_epsilon(href[i], static_cast<T>(i + 1)));
    }

    ac::comm::async_halo_exchange_task<T, ac::mr::device_allocator> he{local_mm,
                                                                       local_nn,
                                                                       local_nn_offset,
                                                                       1};

    he.launch(cart_comm, {distr_dref.get()});

    const size_t nsteps{1000};
    if (rank == 0)
        PRINT_LOG_INFO("Running %zu steps to reach a stable state. Reduce problems size if the "
                       "test "
                       "takes too long. Increase nsteps if stability has not been reached but the "
                       "implementation is correct.",
                       nsteps);

    for (size_t i{0}; i < nsteps; ++i) {
        const ac::shape      nk{2 * rr + as<uint64_t>(1)};
        ac::host_ndbuffer<T> kernel{nk, 1};

        ac::host_ndbuffer<T> distr_dref_tmp{local_mm};
        he.wait({distr_dref.get()});
#if defined(ACM_DEVICE_ENABLED)
        PRINT_LOG_WARNING("Device xcorr and transform not yet implemented");
#else
        ac::xcorr(local_mm,
                  local_nn,
                  local_nn_offset,
                  distr_dref.get(),
                  nk,
                  kernel.get(),
                  distr_dref_tmp.get());
        ac::transform(
            distr_dref_tmp.get(),
            [&nk](const auto& elem) { return elem / prod(nk); },
            distr_dref.get());
#endif
        he.launch(cart_comm, {distr_dref.get()});
    }
    he.wait({distr_dref.get()});

    std::fill(href.begin(), href.end(), -1);
    ac::mpi::gather_advanced(cart_comm,
                             ac::mpi::get_dtype<T>(),
                             local_mm,
                             local_nn,
                             local_nn_offset,
                             distr_dref.data(),
                             global_nn,
                             ac::make_index(global_nn.size(), 0),
                             href.data());

    if (rank == 0) {
        const double expected_value{(prod(global_nn) + 1) / 2.};
        std::cout << "Expected: " << expected_value << std::endl;
        std::cout << "Measured: " << href[0] << std::endl;
        for (size_t i{0}; i < prod(global_nn); ++i)
            ERRCHK(within_machine_epsilon(href[i], expected_value));

        href.display();
        PRINT_LOG_WARNING("Device buffer pipeline not yet tested");
    }
}

int
main()
{
    ac::mpi::init_funneled();
    try {
        const uint64_t nprocs{as<uint64_t>(ac::mpi::get_size(MPI_COMM_WORLD))};
        {
            const ac::shape global_nn{128, 128, 128};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_reduce_axis<ac::mr::host_allocator>(cart_comm, global_nn);
            test_reduce_axis<ac::mr::pinned_host_allocator>(cart_comm, global_nn);
            test_reduce_axis<ac::mr::pinned_write_combined_host_allocator>(cart_comm, global_nn);
            test_reduce_axis<ac::mr::device_allocator>(cart_comm, global_nn);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{4 * nprocs};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{8, 4 * nprocs};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{8, 4, 2 * nprocs};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};

            test_scatter_gather(cart_comm, global_nn);

            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{7, 3 * nprocs};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
            test_scatter_gather_advanced(cart_comm, global_nn);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{4, 5};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
            test_mpi_pack(cart_comm, global_nn);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{8};
            const ac::index rr{1};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
            test_collective_io(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{8};
            const ac::index rr{1};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
            test_pack_transform_reduce(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
        {
            const ac::shape global_nn{6, 4, 8};
            const ac::index rr{2, 1, 3};
            MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
            test_pipeline(cart_comm, global_nn, rr);
            ac::mpi::cart_comm_destroy(&cart_comm);
        }
    }
    catch (const std::exception& e) {
        PRINT_LOG_ERROR("Exception caught");
        ac::mpi::abort();
        return EXIT_FAILURE;
    }
    ac::mpi::finalize();
    return EXIT_SUCCESS;
}
