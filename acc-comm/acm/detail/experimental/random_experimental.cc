#include "random_experimental.h"

#include <algorithm>
#include <random>

#include "acm/detail/type_conversion.h"

namespace acm::experimental {

void
randomize(ac::host_view<double> ptr)
{
    std::generate(ptr.begin(), ptr.end(), []() {
        static std::default_random_engine       gen{12345};
        static std::uniform_real_distribution<> dist{-1, 1};
        return dist(gen);
    });
}

void
randomize(ac::host_view<uint64_t> ptr)
{
    std::generate(ptr.begin(), ptr.end(), []() {
        static std::default_random_engine      gen{12345};
        static std::uniform_int_distribution<> dist{0};
        return as<uint64_t>(dist(gen));
    });
}

} // namespace acm::experimental
