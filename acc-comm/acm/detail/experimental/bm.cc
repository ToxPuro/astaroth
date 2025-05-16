#include "bm.h"

#include "acm/detail/math_utils.h"
#include "acm/detail/timer.h"

namespace bm {

std::vector<std::chrono::steady_clock::duration>
benchmark(const std::function<void()>& init, const std::function<void()>& bench,
          const std::function<void()>& sync, const size_t nsamples)
{
    ac::timer                                        timer;
    std::vector<std::chrono::steady_clock::duration> samples;
    for (size_t i{0}; i < nsamples; ++i) {
        init();
        sync();
        timer.reset();
        sync();
        bench();
        sync();
        samples.push_back(timer.lap());
    }

    return samples;
}

} // namespace bm
