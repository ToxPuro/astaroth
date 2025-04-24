#include "bm.h"

#include "acm/detail/math_utils.h"
#include "acm/detail/timer.h"

namespace bm {

std::vector<std::chrono::nanoseconds::rep>
benchmark(const std::function<void()>& init, const std::function<void()>& bench,
          const std::function<void()>& sync, const size_t nsamples)
{
    // Warmup
    init();
    for (size_t i{0}; i < 3; ++i)
        bench();
    sync();

    // Benchmark
    ac::timer                                  timer;
    std::vector<std::chrono::nanoseconds::rep> samples;
    for (size_t i{0}; i < nsamples; ++i) {
        init();
        sync();
        timer.reset();
        sync();
        bench();
        sync();
        samples.push_back(timer.lap_ns());
    }
    return samples;
}

} // namespace bm
