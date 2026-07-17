#pragma once

#include "datatypes.h"
#include <random>
#include <stdint.h>

void seed_rng(uint32_t seed);

std::mt19937& get_rng();

AcReal random_uniform_real_01(void);