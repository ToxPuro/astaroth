#include "print.h"

#include <inttypes.h>
#include <stdio.h>

#include "math_utils.h"
#include "nalloc.h"

// Prototypes
#define DEFINE_GENERIC_FUNCTION_PRINT(type, format)                                                \
    void print_##type(const char* label, const type value)                                         \
    {                                                                                              \
        printf("%s: " format "\n", label, value);                                                  \
    }

#define DEFINE_GENERIC_FUNCTION_PRINT_ARRAY(type, format)                                          \
    void print_array_##type(const char* label, const size_t count, const type* array)              \
    {                                                                                              \
        ERRCHK(array != NULL);                                                                     \
        printf("%s: {", label);                                                                    \
        for (size_t i = 0; i < count; ++i) {                                                       \
            printf(format, array[i]);                                                              \
            printf("%s", i + 1 < count ? ", " : "");                                               \
        }                                                                                          \
        printf("}\n");                                                                             \
    }

#define DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY_RECURSIVE(type, format)                              \
    static void print_ndarray_recursive_##type(const size_t ndims, const size_t* dims,             \
                                               const type* array)                                  \
    {                                                                                              \
        if (ndims == 1) {                                                                          \
            for (size_t i = 0; i < dims[0]; ++i) {                                                 \
                const size_t len          = 128;                                                   \
                const int print_alignment = 3;                                                     \
                char* str;                                                                         \
                nalloc(len, str);                                                                  \
                snprintf(str, len, format, array[i]);                                              \
                printf("%*s ", print_alignment, str);                                              \
                ndealloc(str);                                                                     \
            }                                                                                      \
            printf("\n");                                                                          \
        }                                                                                          \
        else {                                                                                     \
            const size_t offset = prod(ndims - 1, dims);                                           \
            for (size_t i = 0; i < dims[ndims - 1]; ++i) {                                         \
                if (ndims > 4)                                                                     \
                    printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);                     \
                if (ndims == 4)                                                                    \
                    printf("Cube %zu:\n", i);                                                      \
                if (ndims == 3)                                                                    \
                    printf("Layer %zu:\n", i);                                                     \
                if (ndims == 2)                                                                    \
                    printf("Row %zu: ", i);                                                        \
                print_ndarray_recursive_##type(ndims - 1, dims, &array[i * offset]);               \
            }                                                                                      \
            printf("\n");                                                                          \
        }                                                                                          \
    }

#define DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY(type)                                                \
    void print_ndarray_##type(const char* label, const size_t ndims, const size_t* dims,           \
                              const type* array)                                                   \
    {                                                                                              \
        ERRCHK(array != NULL);                                                                     \
        printf("%s:\n", label);                                                                    \
        print_ndarray_recursive_##type(ndims, dims, array);                                        \
    }

#define DEFINE_GENERIC_FUNCTIONS(type, format)                                                     \
    DEFINE_GENERIC_FUNCTION_PRINT(type, format)                                                    \
    DEFINE_GENERIC_FUNCTION_PRINT_ARRAY(type, format)                                              \
    DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY_RECURSIVE(type, format)                                  \
    DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY(type)

// Definitions
DEFINE_GENERIC_FUNCTIONS(size_t, "%zu")
DEFINE_GENERIC_FUNCTIONS(int64_t, "%" PRId64)
DEFINE_GENERIC_FUNCTIONS(int, "%d")
DEFINE_GENERIC_FUNCTIONS(double, "%g")

// Cleanup
#undef DEFINE_GENERIC_FUNCTIONS
#undef DEFINE_GENERIC_FUNCTION_PRINT
#undef DEFINE_GENERIC_FUNCTION_PRINT_ARRAY
#undef DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY_RECURSIVE
#undef DEFINE_GENERIC_FUNCTION_PRINT_NDARRAY
