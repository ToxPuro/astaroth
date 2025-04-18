#include "bm.h"

#include "acm/detail/math_utils.h"
#include "acm/detail/timer.h"

double
benchmark(const std::string label, const std::function<void()>& init,
          const std::function<void()>& bench, const std::function<void()>& sync)
{
    constexpr size_t nsamples{100};
    ERRCHK(nsamples > 0);

    // Warmup
    init();
    bench();

    // Benchmark
    std::vector<long> samples;
    for (size_t i{0}; i < nsamples; ++i) {
        init();
        sync();
        const auto start{std::chrono::system_clock::now()};
        bench();
        sync();
        const auto elapsed{std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start)};
        samples.push_back(elapsed.count());
    }

    std::sort(samples.begin(), samples.end());
    std::cout << label << ":" << std::endl;
    std::cout << "\tMin: " << samples[0] << " us" << std::endl;
    std::cout << "\tMedian: " << bm::median(samples) << " us" << std::endl;
    std::cout << "\tMax: " << samples[samples.size() - 1] << " us" << std::endl;
    return bm::median(samples);
}

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
