#pragma once
#include <functional>
#include <random>

#include "acm/detail/buffer.h"

template <typename Allocator>
static void
randomize(ac::mr::pointer<double, Allocator> ptr)
{
    ac::host_buffer<double> ref{ptr.size()};

    std::generate(ref.begin(), ref.end(), []() {
        static std::default_random_engine       gen{12345};
        static std::uniform_real_distribution<> dist{-1, 1};
        return dist(gen);
    });

    ac::mr::copy(ref.get(), ptr);
}

/**
 * Returns the  median running time in microseconds.
 * init: function to initialize (e.g. randomizing the inputs)
 * bench: function to benchmark
 */
double benchmark(const std::string label, const std::function<void()>& init,
                 const std::function<void()>& bench, const std::function<void()>& sync);

namespace bm {

/**
 * Benchmark and return a list of samples in ns
 * init:  function that initializes the inputs (e.g. randomize)
 * bench: function that runs the operations to benchmark
 * sync:  function that synchronizes init and bench between iterations
 */
std::vector<long> benchmark(const std::function<void()>& init, const std::function<void()>& bench,
                            const std::function<void()>& sync, const size_t nsamples = 100);

} // namespace bm
