#pragma once

#include <stdint.h>

#include <random>

#include "acreal.h"

void seed_rng(uint32_t seed);

std::mt19937& get_rng();

AcReal random_uniform_real_01(void);