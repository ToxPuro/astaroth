#include "bm.h"
#include "acm/detail/math_utils.h"

template <typename T>
static double
median(const std::vector<T>& vec)
{
    if (vec.size() % 2 == 0) {
        return 0.5 * (vec[(vec.size() - 1) / 2] + vec[vec.size() / 2]);
    }
    else {
        return vec[vec.size() / 2];
    }
}

static int
test_median()
{
    ERRCHK(within_machine_epsilon(median(std::vector<uint64_t>{1, 2, 3}), 2.));
    ERRCHK(within_machine_epsilon(median(std::vector<int>{1, 2, 3, 4}), 2.5));
    ERRCHK(within_machine_epsilon(median(std::vector<int>{1, 2, 3, 4, 5}), 3.));
    ERRCHK(within_machine_epsilon(median(std::vector<int>{1, 2, 3, 4, 5, 6}), 3.5));

    return 0;
}

double
benchmark(const std::string label, const std::function<void()>& init,
          const std::function<void()>& bench)
{
    test_median();

    constexpr size_t nsamples{100};
    ERRCHK(nsamples > 0);

    // Warmup
    init();
    bench();

    // Benchmark
    std::vector<long> samples;
    for (size_t i{0}; i < nsamples; ++i) {
        init();
        const auto start{std::chrono::system_clock::now()};
        bench();
        const auto elapsed{std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start)};
        samples.push_back(elapsed.count());
    }

    std::sort(samples.begin(), samples.end());
    std::cout << label << ":" << std::endl;
    std::cout << "\tMin: " << samples[0] << " us" << std::endl;
    std::cout << "\tMedian: " << median(samples) << " us" << std::endl;
    std::cout << "\tMax: " << samples[samples.size() - 1] << " us" << std::endl;
    return median(samples);
}
