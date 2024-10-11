#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Prototypes
#define DECLARE_GENERIC_FUNCTION_PRINT(type) void print_##type(const char* label, const type value);

#define DECLARE_GENERIC_FUNCTION_PRINT_ARRAY(type)                                                 \
    void print_array_##type(const char* label, const size_t count, const type* array);

#define DECLARE_GENERIC_FUNCTION_PRINT_NDARRAY(type)                                               \
    void print_ndarray_##type(const char* label, const size_t ndims, const size_t* dims,           \
                              const type* array);

#define DECLARE_GENERIC_FUNCTIONS(type)                                                            \
    DECLARE_GENERIC_FUNCTION_PRINT(type)                                                           \
    DECLARE_GENERIC_FUNCTION_PRINT_ARRAY(type)                                                     \
    DECLARE_GENERIC_FUNCTION_PRINT_NDARRAY(type)

// Declarations
DECLARE_GENERIC_FUNCTIONS(size_t)
DECLARE_GENERIC_FUNCTIONS(int64_t)
DECLARE_GENERIC_FUNCTIONS(int)
DECLARE_GENERIC_FUNCTIONS(double)

// Generics
#define print(label, value)                                                                        \
    _Generic((value),                                                                              \
        size_t: print_size_t,                                                                      \
        int64_t: print_int64_t,                                                                    \
        int: print_int,                                                                            \
        double: print_double)(label, value)

#define print_array(label, count, array)                                                           \
    _Generic((array[0]),                                                                           \
        size_t: print_array_size_t,                                                                \
        int64_t: print_array_int64_t,                                                              \
        int: print_array_int,                                                                      \
        double: print_array_double)(label, count, array)

#define print_ndarray(label, ndims, dims, array)                                                   \
    _Generic((array[0]),                                                                           \
        size_t: print_ndarray_size_t,                                                              \
        int64_t: print_ndarray_int64_t,                                                            \
        int: print_ndarray_int,                                                                    \
        double: print_ndarray_double)(label, ndims, dims, array)

#define printd(x) print(#x, (x))
#define printd_array(count, arr) print_array(#arr, (count), (arr))
#define printd_ndarray(ndims, dims, arr) print_ndarray(#arr, (ndims), (dims), (arr))

// Cleanup
#undef DECLARE_GENERIC_FUNCTIONS
#undef DECLARE_GENERIC_FUNCTION_PRINT
#undef DECLARE_GENERIC_FUNCTION_PRINT_ARRAY
#undef DECLARE_GENERIC_FUNCTION_PRINT_NDARRAY
