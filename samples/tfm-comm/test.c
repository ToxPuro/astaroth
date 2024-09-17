#include "test.h"

#include <stdio.h>
#include <stdlib.h>

// #include "errchk.h"
#include "math_utils.h"
// #include "print.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

// #define PRINT_TYPE_DECLARATION(type) void print_type_##type(const type value)

// #define PRINT_TYPE(type, specifier)                                                                \
//     PRINT_TYPE_DECLARATION(type) { printf(specifier, value); }

// #define PRINT_LABELED_TYPE_DECLARATION(type) void print_##type(const char* label, const type
// value)

// #define PRINT_LABELED_TYPE(type)                                                                   \
//     PRINT_LABELED_TYPE_DECLARATION(type)                                                           \
//     {                                                                                              \
//         printf("%s: ", label);                                                                     \
//         print_type_##type(value);                                                                  \
//         printf("\n");                                                                              \
//     }

// #define PRINT_LABELED_ARRAY_DECLARATION(type)                                                      \
//     void print_array_##type(const char* label, const size_t count, const type* arr)

// #define PRINT_LABELED_ARRAY(type)                                                                  \
//     PRINT_LABELED_ARRAY_DECLARATION(type)                                                          \
//     {                                                                                              \
//         printf("%s: (", label);                                                                    \
//         for (size_t i = 0; i < count; ++i) {                                                       \
//             print_type_##type(arr[i]);                                                             \
//             printf("%s", i < count - 1 ? ", " : "");                                               \
//         }                                                                                          \
//         printf(")\n");                                                                             \
//     }

// #define GEN_PRINT_FUNCTION_DECLARATIONS(type)                                                      \
//     PRINT_TYPE_DECLARATION(type);                                                                  \
//     PRINT_LABELED_TYPE_DECLARATION(type);                                                          \
//     PRINT_LABELED_ARRAY_DECLARATION(type);

// #define GEN_PRINT_FUNCTION_DEFINITIONS(type, specifier)                                            \
//     PRINT_TYPE(type, specifier)                                                                    \
//     PRINT_LABELED_TYPE(type)                                                                       \
//     PRINT_LABELED_ARRAY(type)

// // GEN_PRINT_FUNCTION_DECLARATIONS(int)

// GEN_PRINT_FUNCTION_DEFINITIONS(int, "%d")
// GEN_PRINT_FUNCTION_DEFINITIONS(int64_t, "%lld")
// GEN_PRINT_FUNCTION_DEFINITIONS(size_t, "%zu")
// GEN_PRINT_FUNCTION_DEFINITIONS(double, "%lg")

// #define GENERIC(type, fn)                                                                          \
//     type:                                                                                          \
//     fn##_##type

// #define print(label, value) _Generic((value), GENERIC(int, print))(label, value)

int
main(void)
{
    printf("Testing...\n");
    test_math_utils();
    printf("Complete\n");
    print_double("aa", 1.0);
    print_double("label", 1.0);
    double arr[3] = {1, 2, 3};
    print_array_double("dddd", 3, arr);
    print("Test", 1);
    return EXIT_SUCCESS;
}
