#pragma once

#include <stdio.h>

static const char type_specifier_size_t[]  = "%zu";
static const char type_specifier_int[]     = "%d";
static const char type_specifier_int64_t[] = "%lld";
static const char type_specifier_double[]  = "%g";

#define CREATE_PRINT_FUNCTION(type)                                                                \
    static inline void print_##type(const char* label, const type value)                           \
    {                                                                                              \
        printf("%s: ", label);                                                                     \
        printf(type_specifier_##type, value);                                                      \
        printf("\n");                                                                              \
    }

#define CREATE_PRINT_ARRAY_FUNCTION(type)                                                          \
    static inline void print_array_##type(const char* label, const size_t count, const type* arr)  \
    {                                                                                              \
        printf("%s: (", label);                                                                    \
        for (size_t i = 0; i < count; ++i) {                                                       \
            printf(type_specifier_##type, arr[i]);                                                 \
            printf("%s", i < count - 1 ? ", " : "");                                               \
        }                                                                                          \
        printf(")\n");                                                                             \
    }

static inline size_t
print_prod_todo_remove(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

#define CREATE_PRINT_NDARRAY_RECURSIVE_FUNCTION(type)                                              \
    static inline void print_ndarray_recursive_##type(const size_t ndims, const size_t* dims,      \
                                                      const type* arr)                             \
    {                                                                                              \
        if (ndims == 1) {                                                                          \
            for (size_t i = 0; i < dims[0]; ++i) {                                                 \
                const size_t len          = 128;                                                   \
                const int print_alignment = 3;                                                     \
                char str[len];                                                                     \
                snprintf(str, len, type_specifier_##type, arr[i]);                                 \
                printf("%*s ", print_alignment, str);                                              \
            }                                                                                      \
            printf("\n");                                                                          \
        }                                                                                          \
        else {                                                                                     \
            const size_t offset = print_prod_todo_remove(ndims - 1, dims);                         \
            for (size_t i = 0; i < dims[ndims - 1]; ++i) {                                         \
                if (ndims > 4)                                                                     \
                    printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);                     \
                if (ndims == 4)                                                                    \
                    printf("Cube %zu:\n", i);                                                      \
                if (ndims == 3)                                                                    \
                    printf("Layer %zu:\n", i);                                                     \
                if (ndims == 2)                                                                    \
                    printf("Row %zu:", i);                                                         \
                print_ndarray_recursive_##type(ndims - 1, dims, &arr[i * offset]);                 \
            }                                                                                      \
            printf("\n");                                                                          \
        }                                                                                          \
    }

#define CREATE_PRINT_NDARRAY_FUNCTION(type)                                                        \
    static inline void print_ndarray_##type(const char* label, const size_t ndims,                 \
                                            const size_t* dims, const type* arr)                   \
    {                                                                                              \
        printf("%s:\n", label);                                                                    \
        print_ndarray_recursive_##type(ndims, dims, arr);                                          \
    }

CREATE_PRINT_FUNCTION(size_t)
CREATE_PRINT_ARRAY_FUNCTION(size_t)
CREATE_PRINT_NDARRAY_RECURSIVE_FUNCTION(size_t)
CREATE_PRINT_NDARRAY_FUNCTION(size_t)

CREATE_PRINT_FUNCTION(int)
CREATE_PRINT_ARRAY_FUNCTION(int)
CREATE_PRINT_NDARRAY_RECURSIVE_FUNCTION(int)
CREATE_PRINT_NDARRAY_FUNCTION(int)

CREATE_PRINT_FUNCTION(int64_t)
CREATE_PRINT_ARRAY_FUNCTION(int64_t)
CREATE_PRINT_NDARRAY_RECURSIVE_FUNCTION(int64_t)
CREATE_PRINT_NDARRAY_FUNCTION(int64_t)

CREATE_PRINT_FUNCTION(double)
CREATE_PRINT_ARRAY_FUNCTION(double)
CREATE_PRINT_NDARRAY_RECURSIVE_FUNCTION(double)
CREATE_PRINT_NDARRAY_FUNCTION(double)

#define print(label, value)                                                                        \
    _Generic((value),                                                                              \
        size_t: print_size_t,                                                                      \
        int64_t: print_int64_t,                                                                    \
        int: print_int,                                                                            \
        double: print_double)(label, value)

#define print_array(label, count, arr)                                                             \
    _Generic((arr),                                                                                \
        size_t *: print_array_size_t,                                                              \
        int64_t *: print_array_int64_t,                                                            \
        int*: print_array_int,                                                                     \
        double*: print_array_double,                                                               \
        const size_t*: print_array_size_t,                                                         \
        const int64_t*: print_array_int64_t,                                                       \
        const int*: print_array_int,                                                               \
        const double*: print_array_double)(label, count, arr)
