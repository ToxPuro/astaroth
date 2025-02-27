#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>

#include "acm/detail/errchk.h"
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

int
main()
{
    ac::mpi::init_funneled();
    try {

        // Setup domain and communicator
        const ac::shape global_nn{256, 256, 256};
        MPI_Comm        cart_comm{ac::mpi::cart_comm_create(MPI_COMM_WORLD, global_nn)};
        const auto      rr{ac::make_index(global_nn.size(), 3)};

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

        // MPI
        randomize(href.get());
        benchmark("MPI htoh pack", [&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          href.data(),
                          hpack.size(),
                          hpack.data());
        });

        randomize(href.get());
        benchmark("MPI htod pack", [&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          href.data(),
                          dpack.size(),
                          dpack.data());
        });

        randomize(dref.get());
        benchmark("MPI dtoh pack", [&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          dref.data(),
                          hpack.size(),
                          hpack.data());
        });

        randomize(dref.get());
        benchmark("MPI dtod pack", [&]() {
            ac::mpi::pack(cart_comm,
                          ac::mpi::get_dtype<T>(),
                          local_mm,
                          local_nn,
                          rr,
                          dref.data(),
                          dpack.size(),
                          dpack.data());
        });

        randomize(dref.get());
        benchmark("MPI dtod unpack", [&]() {
            ac::mpi::unpack(cart_comm,
                            ac::mpi::get_dtype<T>(),
                            dpack.size(),
                            dpack.data(),
                            local_mm,
                            local_nn,
                            rr,
                            dref.data());
        });

        randomize(dref.get());
        benchmark("ACM dtod pack",
                  [&]() { pack(local_mm, local_nn, rr, std::vector{dref.get()}, dpack.get()); });

        randomize(dpack.get());
        benchmark("ACM dtod unpack",
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
