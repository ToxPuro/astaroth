#pragma once
#include <stddef.h>
#include <stdint.h>

#define print(label, value)                                                                        \
    _Generic((value), size_t: print_size_t, int64_t: print_int64_t, int: print_int)(label, value)

#define print_array(label, count, arr)                                                             \
    _Generic((arr),                                                                                \
        size_t *: print_size_t_array,                                                              \
        int64_t *: print_int64_t_array,                                                            \
        int*: print_int_array)(label, count, arr)

void print_size_t(const char* label, const size_t value);

void print_int64_t(const char* label, const int64_t value);

void print_int(const char* label, const int value);

void print_size_t_array(const char* label, const size_t count, const size_t arr[]);

void print_int64_t_array(const char* label, const size_t count, const int64_t arr[]);

void print_int_array(const char* label, const size_t count, const int arr[]);
