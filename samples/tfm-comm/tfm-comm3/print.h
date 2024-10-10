#pragma once
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#define FORMAT_SPECIFIER(value)                                                                    \
    _Generic((value), size_t: "%zu", float: "%f", double: "%g", int: "%d", int64_t: "%" PRId64)

#define PRINT(value)                                                                               \
    do {                                                                                           \
        printf("%s: ", #value);                                                                    \
        printf(FORMAT_SPECIFIER((value)), (value));                                                \
        printf("\n");                                                                              \
    } while (0)

#define PRINT_ARRAY(count, arr)                                                                    \
    do {                                                                                           \
        printf("%s: ", #arr);                                                                      \
        for (size_t i_ = 0; i_ < count; ++i_) {                                                    \
            printf(FORMAT_SPECIFIER((arr)[0]), (arr)[0]);                                          \
            printf(" ");                                                                           \
        }                                                                                          \
        printf("\n");                                                                              \
    } while (0)

#define PRINT_BUFFER(buffer) PRINT_ARRAY(buffer.count, buffer.data)

#define PRINT_SHAPE(shape)                                                                         \
    do {                                                                                           \
        PRINT(shape.ndims);                                                                        \
        PRINT_ARRAY(shape.ndims, shape.dims);                                                      \
        PRINT_ARRAY(shape.ndims, shape.subdims);                                                   \
        PRINT_ARRAY(shape.ndims, shape.offset);                                                    \
    } while (0)
