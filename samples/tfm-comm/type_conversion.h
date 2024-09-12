#pragma once
#include <stddef.h>
#include <stdint.h>

size_t as_size_t(const int64_t i);

int64_t as_int64_t(const size_t i);

void as_size_t_array(const size_t count, const int64_t* a, size_t* b);

void as_int64_t_array(const size_t count, const size_t* a, int64_t* b);