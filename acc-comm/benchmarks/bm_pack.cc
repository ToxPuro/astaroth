#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>

#include "acm/detail/errchk.h"
#include "acm/detail/math_utils.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ndbuffer.h"
#include "acm/detail/pack.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/print_debug.h"

static void
benchmark(const std::string label, const std::function<void()>& fn)
{
    constexpr size_t nsamples{10};
    ERRCHK(nsamples > 0);

    // Warmup
    fn();

    // Benchmark
    std::vector<long> samples;
    for (size_t i{0}; i < nsamples; ++i) {
        const auto start{std::chrono::system_clock::now()};
        fn();
        const auto elapsed{std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start)};
        samples.push_back(elapsed.count());
    }

    std::sort(samples.begin(), samples.end());
    std::cout << label << ":" << std::endl;
    std::cout << "\tMin: " << samples[0] << " us" << std::endl;
    std::cout << "\tMedian: " << samples[samples.size() / 2] << " us" << std::endl;
    std::cout << "\t90th: "
              << samples[static_cast<size_t>(0.5 * static_cast<double>(samples.size() - 1))]
              << " us" << std::endl;
    std::cout << "\tMax: " << samples[samples.size() - 1] << " us" << std::endl;
}

template <typename Allocator>
static void
randomize(ac::mr::pointer<double, Allocator> ptr)
{
    std::default_random_engine       gen;
    std::uniform_real_distribution<> dist{-1, 1};

    ac::host_buffer<double> ref{ptr.size()};
    std::generate(ref.begin(), ref.end(), [&dist, &gen]() { return dist(gen); });

    ac::mr::copy(ref.get(), ptr);
}

/** Verifies the packing function used for benchmarking.
 * Strategy:
 *     1) Setup a full ndbuffer
 *     2) Unpack iota to the position which should be packed
 *     3) Call the pack function
 *     4) Confirm that the result matches iota
 */
template <typename T, typename AllocatorA, typename AllocatorB>
static int
verify_pack(const ac::shape& mm, const ac::shape& nn, const ac::index& rr,
            ac::mr::pointer<T, AllocatorA> input, const std::function<void()>& fn,
            ac::mr::pointer<T, AllocatorB> output)
{
    // Scramble inputs
    randomize(input);
    randomize(output);

    // Test
    ac::host_ndbuffer<T> full{mm, 0};

    ac::host_ndbuffer<T> packed{nn};
    std::iota(packed.begin(), packed.end(), 1);

    unpack(packed.get(), mm, nn, rr, {full.get()});
    std::fill(packed.begin(), packed.end(), 0);
    // full.display();
    // packed.display();

    ac::mr::copy(full.get(), input);
    fn();
    ac::mr::copy(output, packed.get());

    // packed.display();
    for (size_t i{0}; i < packed.size(); ++i)
        ERRCHK(within_machine_epsilon(packed[i], static_cast<T>(i + 1)));

    return 0;
}

int
main()
{
    ac::mpi::init_funneled();
    try {

        // Setup domain and communicator
        const ac::shape global_nn{8, 8};
        MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const auto      rr{ac::make_index(global_nn.size(), 1)};

        // Setup the buffers
        const ac::shape local_mm{ac::mpi::get_local_mm(cart_comm, global_nn, rr)};
        const ac::shape local_nn{ac::mpi::get_local_nn(cart_comm, global_nn)};

        using T = double;
        ac::ndbuffer<T, ac::mr::host_allocator>   href{local_mm};
        ac::ndbuffer<T, ac::mr::host_allocator>   htst{local_mm};
        ac::ndbuffer<T, ac::mr::host_allocator>   hpack{local_nn};
        ac::ndbuffer<T, ac::mr::device_allocator> dref{local_mm};
        ac::ndbuffer<T, ac::mr::device_allocator> dtst{local_mm};
        ac::ndbuffer<T, ac::mr::device_allocator> dpack{local_nn};

        // verify_pack(
        //     local_mm,
        //     local_nn,
        //     rr,
        //     href.get(),
        //     [&]() {
        //         ac::mpi::pack(cart_comm,
        //                       ac::mpi::get_dtype<T>(),
        //                       local_mm,
        //                       local_nn,
        //                       rr,
        //                       href.data(),
        //                       hpack.size(),
        //                       hpack.data());
        //     },
        //     hpack.get());
        // return EXIT_SUCCESS;

        // MPI
        auto mpi_htoh_pack{[&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          href.data(),
                          hpack.size(),
                          hpack.data());
        }};
        verify_pack(local_mm, local_nn, rr, href.get(), mpi_htoh_pack, hpack.get());
        randomize(href.get());
        benchmark("MPI htoh pack", mpi_htoh_pack);

        auto mpi_htod_pack{[&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          href.data(),
                          dpack.size(),
                          dpack.data());
        }};
        verify_pack(local_mm, local_nn, rr, href.get(), mpi_htod_pack, dpack.get());
        randomize(href.get());
        benchmark("MPI htod pack", mpi_htod_pack);

        auto mpi_dtoh_pack{[&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          dref.data(),
                          hpack.size(),
                          hpack.data());
        }};
        verify_pack(local_mm, local_nn, rr, dref.get(), mpi_dtoh_pack, hpack.get());
        randomize(dref.get());
        benchmark("MPI dtoh pack", mpi_dtoh_pack);

        auto mpi_dtod_pack{[&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          dref.data(),
                          dpack.size(),
                          dpack.data());
        }};
        verify_pack(local_mm, local_nn, rr, dref.get(), mpi_dtod_pack, dpack.get());
        randomize(dref.get());
        benchmark("MPI dtod pack", mpi_dtod_pack);

        randomize(dref.get());
        benchmark("MPI dtod unpack (note: not verified)", [&]() {
            ac::mpi::unpack(cart_comm,
                            ac::mpi::get_dtype<T>(),
                            dpack.size(),
                            dpack.data(),
                            local_mm,
                            local_nn,
                            rr,
                            dref.data());
        });

        auto acm_dtod_pack{
            [&]() { pack(local_mm, local_nn, rr, std::vector{dref.get()}, dpack.get()); }};
        verify_pack(local_mm, local_nn, rr, dref.get(), acm_dtod_pack, dpack.get());
        randomize(dref.get());
        benchmark("ACM dtod pack", acm_dtod_pack);

        randomize(dpack.get());
        benchmark("ACM dtod unpack (note: not verified)",
                  [&]() { unpack(dpack.get(), local_mm, local_nn, rr, std::vector{dref.get()}); });

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
