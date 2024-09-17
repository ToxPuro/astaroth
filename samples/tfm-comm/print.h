#pragma once
#include <stddef.h>
#include <stdint.h>

#define print_type(value)                                                                          \
    _Generic((value),                                                                              \
        size_t: print_type_size_t,                                                                 \
        int64_t: print_type_int64_t,                                                               \
        int: print_type_int,                                                                       \
        double: print_type_double)(value)

void print_type_size_t(const size_t value);
void print_type_int64_t(const int64_t value);
void print_type_int(const int value);
void print_type_double(const double value);

#define print(label, value)                                                                        \
    _Generic((value),                                                                              \
        size_t: print_size_t,                                                                      \
        int64_t: print_int64_t,                                                                    \
        int: print_int,                                                                            \
        double: print_double)(label, value)

void print_size_t(const char* label, const size_t value);
void print_int64_t(const char* label, const int64_t value);
void print_int(const char* label, const int value);
void print_double(const char* label, const double value);

#define print_array(label, count, arr)                                                             \
    _Generic((arr),                                                                                \
        size_t *: print_array_size_t,                                                              \
        int64_t *: print_array_int64_t,                                                            \
        int*: print_array_int,                                                                     \
        double*: print_array_double)(label, count, arr)

void print_array_size_t(const char* label, const size_t count, const size_t* arr);
void print_array_int64_t(const char* label, const size_t count, const int64_t* arr);
void print_array_int(const char* label, const size_t count, const int* arr);
void print_array_double(const char* label, const size_t count, const double* arr);
