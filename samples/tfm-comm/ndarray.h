#pragma once
#include <stdbool.h>
#include <stddef.h>

void set_ndarray(const size_t value, const size_t ndims, const size_t* start, const size_t* subdims,
                 const size_t* dims, size_t* arr);

/** Checks whether all of the `count` elements starting from start_a and start_b are equal */
bool ndarray_equals(const size_t count, const size_t ndims, const size_t* a_offset,
                    const size_t* b_offset, const size_t* dims, const size_t* arr);

void print_ndarray(const char* label, const size_t ndims, const size_t* dims, const size_t* arr);

void ndarray_test(void);
