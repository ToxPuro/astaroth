#include "random_experimental.h"

#include <random>
#include <algorithm>

#include "acm/detail/type_conversion.h"

namespace acm::experimental {

void
randomize(ac::mr::host_pointer<double> ptr)
{
    std::generate(ptr.begin(), ptr.end(), []() {
        static std::default_random_engine       gen{12345};
        static std::uniform_real_distribution<> dist{-1, 1};
        return dist(gen);
    });
}

void
randomize(ac::mr::host_pointer<uint64_t> ptr)
{
    std::generate(ptr.begin(), ptr.end(), []() {
        static std::default_random_engine       gen{12345};
        static std::uniform_int_distribution<> dist{0};
        return as<uint64_t>(dist(gen));
    });
}

}
