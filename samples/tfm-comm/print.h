#pragma once
#include <stddef.h>
#include <stdint.h>

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

// Cleanup
#undef DECLARE_GENERIC_FUNCTIONS
#undef DECLARE_GENERIC_FUNCTION_PRINT
#undef DECLARE_GENERIC_FUNCTION_PRINT_ARRAY
#undef DECLARE_GENERIC_FUNCTION_PRINT_NDARRAY

static inline void
print_matrix(const char* label, const size_t nrows, const size_t ncols, const size_t* matrix)
{
    print_ndarray(label, 2, ((size_t[]){ncols, nrows}), matrix);
}
