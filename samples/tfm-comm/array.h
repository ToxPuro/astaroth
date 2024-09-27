#pragma once
#include <stdbool.h>
#include <stddef.h>

// Macros
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

void copy(const size_t count, const size_t* in, size_t* out);

void copyi(const size_t count, const int* in, int* out);

void test_array(void);
