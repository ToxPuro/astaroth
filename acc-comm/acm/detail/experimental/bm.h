#pragma once
#include <chrono>
#include <functional>
#include <type_traits>

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

template <typename T>
double
median(const std::vector<std::chrono::steady_clock::duration>& vec)
{
    // Enabled only for std::chrono time units
    static_assert(std::is_same_v<T, std::chrono::nanoseconds> ||
                  std::is_same_v<T, std::chrono::microseconds> ||
                  std::is_same_v<T, std::chrono::milliseconds> ||
                  std::is_same_v<T, std::chrono::seconds>);

    std::vector<typename T::rep> values;
    for (const auto& elem : vec)
        values.push_back(std::chrono::duration_cast<T>(elem).count());

    return median(values);
}

/**
 * Benchmark and return a list of samples in ns
 * init:  function that initializes the inputs (e.g. randomize)
 * bench: function that runs the operations to benchmark
 * sync:  function that synchronizes init and bench between iterations
 */
std::vector<std::chrono::steady_clock::duration> benchmark(const std::function<void()>& init,
                                                           const std::function<void()>& bench,
                                                           const std::function<void()>& sync,
                                                           const size_t                 nsamples);

} // namespace bm
