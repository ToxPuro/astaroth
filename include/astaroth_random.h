#pragma once

#include <stdint.h>

#include <random>

#include "acreal.h"
#include "func_define.h"

AC_BEGIN_C_DECLARATIONS

void seed_rng(uint32_t seed);

AcReal random_uniform_real_01(void);

AC_END_C_DECLARATIONS

std::mt19937& get_rng();
