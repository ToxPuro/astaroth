#pragma once
#include <stddef.h>
#include <stdint.h>

#define as_size_t(x) _Generic((x), int64_t: int64_t_as_size_t, int: int_as_size_t)(x)
#define as_int64_t(x) _Generic((x), size_t: size_t_as_int64_t, int: int_as_int64_t)(x)
#define as_int(x) _Generic((x), size_t: size_t_as_int, int64_t: int64_t_as_int)(x)

size_t int64_t_as_size_t(const int64_t i);
size_t int_as_size_t(const int i);
int64_t size_t_as_int64_t(const size_t i);
int64_t int_as_int64_t(const int i);
int size_t_as_int(const size_t i);
int int64_t_as_int(const int64_t i);

#define as_size_t_array(count, a, b)                                                               \
    _Generic((a),                                                                                  \
        int64_t *: int64_t_as_size_t_array,                                                        \
        const int64_t*: int64_t_as_size_t_array,                                                   \
        int*: int_as_size_t_array,                                                                 \
        const int*: int_as_size_t_array)(count, a, b)
#define as_int64_t_array(count, a, b)                                                              \
    _Generic((a),                                                                                  \
        size_t *: size_t_as_int64_t_array,                                                         \
        const size_t*: size_t_as_int64_t_array,                                                    \
        int*: int_as_int64_t_array,                                                                \
        const int*: int_as_int64_t_array)(count, a, b)
#define as_int_array(count, a, b)                                                                  \
    _Generic((a),                                                                                  \
        size_t *: size_t_as_int_array,                                                             \
        const size_t*: size_t_as_int_array,                                                        \
        int64_t*: int64_t_as_int_array,                                                            \
        const int64_t*: int64_t_as_int_array, )(count, a, b)

void int64_t_as_size_t_array(const size_t count, const int64_t* a, size_t* b);
void int_as_size_t_array(const size_t count, const int* a, size_t* b);
void size_t_as_int64_t_array(const size_t count, const size_t* a, int64_t* b);
void int_as_int64_t_array(const size_t count, const int* a, int64_t* b);
void size_t_as_int_array(const size_t count, const size_t* a, int* b);
void int64_t_as_int_array(const size_t count, const int64_t* a, int* b);
