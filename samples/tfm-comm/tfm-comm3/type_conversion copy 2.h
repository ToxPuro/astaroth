#pragma once
#include <stddef.h>
#include <stdint.h>

double uint64_t_as_double(const uint64_t x);
#define as_double(x) _Generic((x), uint64_t: uint64_t_as_double)(x)

uint64_t uint64_t_as_uint64_t(const uint64_t i);
int int_as_int(const int i);
uint64_t int64_t_as_uint64_t(const int64_t i);
uint64_t int_as_uint64_t(const int i);
int64_t uint64_t_as_int64_t(const uint64_t i);
int64_t int_as_int64_t(const int i);
int uint64_t_as_int(const uint64_t i);
int int64_t_as_int(const int64_t i);

#define as_uint64_t(x)                                                                             \
    _Generic((x),                                                                                  \
        uint64_t: uint64_t_as_uint64_t,                                                            \
        int64_t: int64_t_as_uint64_t,                                                              \
        int: int_as_uint64_t)(x)
#define as_int64_t(x) _Generic((x), uint64_t: uint64_t_as_int64_t, int: int_as_int64_t)(x)
#define as_int(x)                                                                                  \
    _Generic((x), uint64_t: uint64_t_as_int, int64_t: int64_t_as_int, int: int_as_int)(x)

void size_t_as_size_t_array(const size_t count, const size_t* a, size_t* b);
void uint64_t_as_size_t_array(const size_t count, const uint64_t* a, size_t* b);
void int64_t_as_size_t_array(const size_t count, const int64_t* a, size_t* b);
void int_as_size_t_array(const size_t count, const int* a, size_t* b);
void size_t_as_uint64_t_array(const size_t count, const size_t* a, uint64_t* b);
void uint64_t_as_uint64_t_array(const size_t count, const uint64_t* a, uint64_t* b);
void int64_t_as_uint64_t_array(const size_t count, const int64_t* a, uint64_t* b);
void int_as_uint64_t_array(const size_t count, const int* a, uint64_t* b);
void size_t_as_int64_t_array(const size_t count, const size_t* a, int64_t* b);
void uint64_t_as_int64_t_array(const size_t count, const uint64_t* a, int64_t* b);
void int64_t_as_int64_t_array(const size_t count, const int64_t* a, int64_t* b);
void int_as_int64_t_array(const size_t count, const int* a, int64_t* b);
void size_t_as_int_array(const size_t count, const size_t* a, int* b);
void uint64_t_as_int_array(const size_t count, const uint64_t* a, int* b);
void int64_t_as_int_array(const size_t count, const int64_t* a, int* b);
void int_as_int_array(const size_t count, const int* a, int* b);

#define as_size_t_array(count, a, b)                                                               \
    _Generic((a[0]),                                                                               \
        size_t: size_t_as_size_t_array,                                                            \
        uint64_t: uint64_t_as_size_t_array,                                                        \
        int64_t: int64_t_as_size_t_array,                                                          \
        int: int_as_size_t_array)(count, a, b)
#define as_uint64_t_array(count, a, b)                                                             \
    _Generic((a[0]),                                                                               \
        size_t: size_t_as_uint64_t_array,                                                          \
        uint64_t: uint64_t_as_uint64_t_array,                                                      \
        int64_t: int64_t_as_uint64_t_array,                                                        \
        int: int_as_uint64_t_array)(count, a, b)
#define as_int64_t_array(count, a, b)                                                              \
    _Generic((a[0]),                                                                               \
        size_t: size_t_as_int64_t_array,                                                           \
        uint64_t: uint64_t_as_int64_t_array,                                                       \
        int64_t: int64_t_as_int64_t_array,                                                         \
        int: int_as_int64_t_array)(count, a, b)
#define as_int_array(count, a, b)                                                                  \
    _Generic((a[0]),                                                                               \
        size_t: size_t_as_int_array,                                                               \
        uint64_t: uint64_t_as_int_array,                                                           \
        int64_t: int64_t_as_int_array,                                                             \
        int: int_as_int_array)(count, a, b)
