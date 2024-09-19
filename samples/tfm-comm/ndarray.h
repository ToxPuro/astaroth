#pragma once
#include <stddef.h>

void set_ndarray(const size_t value, const size_t ndims, const size_t* start, const size_t* subdims,
                 const size_t* dims, size_t* arr);

void print_ndarray(const size_t ndims, const size_t* dims, const size_t* arr);
