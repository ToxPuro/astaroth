#pragma once
#include <functional>
#include <random>

#include "acm/detail/buffer.h"

/**
 * Returns the  median running time in microseconds.
 * init: function to initialize (e.g. randomizing the inputs)
 * bench: function to benchmark
 */
double benchmark(const std::string label, const std::function<void()>& init,
                 const std::function<void()>& bench, const std::function<void()>& sync);

namespace bm {

template <typename T>
double
median(const std::vector<T>& vec)
{
    if (vec.size() % 2 == 0) {
        return 0.5 * static_cast<double>(vec[(vec.size() - 1) / 2] + vec[vec.size() / 2]);
    }
    else {
        return static_cast<double>(vec[vec.size() / 2]);
    }
}

/**
 * Benchmark and return a list of samples in ns
 * init:  function that initializes the inputs (e.g. randomize)
 * bench: function that runs the operations to benchmark
 * sync:  function that synchronizes init and bench between iterations
 */
std::vector<std::chrono::nanoseconds::rep> benchmark(const std::function<void()>& init,
                                                     const std::function<void()>& bench,
                                                     const std::function<void()>& sync,
                                                     const size_t                 nsamples);

} // namespace bm
