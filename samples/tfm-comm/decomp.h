#pragma once

#include "datatypes.h"

Shape decompose(const Shape& nn, uint64_t nprocs);

void test_decomp(void);
