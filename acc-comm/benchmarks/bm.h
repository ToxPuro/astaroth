#pragma once
#include <functional>
#include <random>

#include "acm/detail/buffer.h"

template <typename Allocator>
static void
randomize(ac::mr::pointer<double, Allocator> ptr)
{
    static ac::host_buffer<double> ref{ptr.size()};

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
