#include "astaroth_random.h"

static std::mt19937 rng(19937123);

void
seed_rng(uint32_t seed)
{
    rng.seed(seed);
}

std::mt19937&
get_rng()
{
    return rng;
}

AcReal
random_uniform_real_01(void)
{
    std::uniform_real_distribution<AcReal> u01(0, 1);
    return u01(get_rng());
}
