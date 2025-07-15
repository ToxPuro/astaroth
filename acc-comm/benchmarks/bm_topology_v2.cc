/*
 * Draft of a simple htod, dtoh, intra-node dtod benchmark.
 * NOTE: unfinished.
 * Use `rocm-smi --shownodesbw` and `rocm-bandwidth-test` instead.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "acm/detail/buffer.h"
#include "acm/detail/errchk_cuda.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/type_conversion.h"

#include "acm/detail/experimental/bm.h"
#include "acm/detail/experimental/random_experimental.h"

#include "acm/detail/errchk_cuda.h"

namespace ac {

template <typename T, typename U, typename A_in, typename A_out, typename B_in, typename B_out>
void
bidirectional_copy(const ac::view<T, A_in>& a_in, const ac::view<U, B_in>& b_in, //
                   ac::view<T, A_out> a_out, ac::view<U, B_out> b_out)
{
    auto a_stream{copy_async(a_in, a_out)};
    auto b_stream{copy_async(b_in, b_out)};

    a_stream.wait();
    b_stream.wait();
}

} // namespace ac

template <typename A, typename B>
static auto
benchmark_bidirectional_bandwidth_gib_s(const size_t nsamples, const size_t bytes)
{
    using T = double;
    ERRCHK_EXPR_DESC(bytes % sizeof(T) == 0,
                     "Parameter `bytes` must be a multiple of %zu",
                     sizeof(T));
    const size_t count{bytes / sizeof(T)};

    ac::buffer<double, A> a_src{count};
    ac::buffer<double, B> a_dst{count};
    ac::buffer<double, B> b_src{count};
    ac::buffer<double, A> b_dst{count};
    acm::experimental::randomize(a_src);
    acm::experimental::randomize(b_src);

    auto init = []() {};
    auto sync = [] {
#if defined(ACM_DEVICE_ENABLED)
        if constexpr (std::is_same_v<A, ac::mr::device_allocator> ||
                      std::is_same_v<B, ac::mr::device_allocator>)
            ERRCHK_CUDA_API(cudaDeviceSynchronize());
#endif
        /* Do nothing */
    };
    auto bench = [&]() {
        ac::bidirectional_copy(a_src.get(), b_src.get(), a_dst.get(), b_dst.get());
    };

    // Benchmark
    const auto times{bm::benchmark(init, bench, sync, nsamples)};

    // Verify
    {
        auto model{a_src.to_host()};
        auto ref{a_dst.to_host()};
        for (size_t i{0}; i < model.size(); ++i)
            ERRCHK(within_machine_epsilon(model[i], ref[i]));
    }
    {
        auto model{b_src.to_host()};
        auto ref{b_dst.to_host()};
        for (size_t i{0}; i < model.size(); ++i)
            ERRCHK(within_machine_epsilon(model[i], ref[i]));
    }

    // Convert to GiB/s
    std::vector<double> results;
    for (const auto& time : times) {
        const auto seconds{std::chrono::duration_cast<std::chrono::nanoseconds>(time).count() /
                           1e9l};
        const auto gib_transferred{2 * bytes / std::pow(1024.0l, 3)};

        results.push_back(gib_transferred / seconds);
    }

    return results;
}

const auto
print(const size_t problem_size, const std::vector<double>& results)
{
    for (size_t i{0}; i < results.size(); ++i) {
        std::cout << problem_size << ",";
        std::cout << i << ",";
        std::cout << results.size() << ",";
        std::cout << results[i] << std::endl;
    }
}

int
main(void)
{
    ac::mpi::init_funneled();
    try {

// Select device
#if defined(ACM_DEVICE_ENABLED)
        int device_count{0};
        ERRCHK_CUDA_API(cudaGetDeviceCount(&device_count));
        ERRCHK_CUDA_API(cudaSetDevice(ac::mpi::get_rank(MPI_COMM_WORLD) % device_count));
#endif

        if (ac::mpi::get_rank(MPI_COMM_WORLD) == 0) {
            constexpr size_t nsamples{5};
            constexpr size_t problem_size{256 * 1024 * 1024}; // Bytes

            const auto results{
                benchmark_bidirectional_bandwidth_gib_s<ac::mr::device_allocator,
                                                        ac::mr::host_allocator>(nsamples,
                                                                                problem_size)};

            print(problem_size, results);
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
