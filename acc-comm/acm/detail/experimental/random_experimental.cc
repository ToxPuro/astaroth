#include "random_experimental.h"

#include <random>

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

}
