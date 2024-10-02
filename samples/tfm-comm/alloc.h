#pragma once
#include "errchk.h"

#include <string.h>

/**
 * Allocate an array
 * alloc(const size_t count, void** ptr)
 */
#define alloc(count, ptr)                                                                          \
    do {                                                                                           \
        (ptr) = malloc(sizeof((ptr)[0]) * (count));                                                \
        ERRCHK((ptr) != NULL);                                                                     \
    } while (0)

/**
 * Deallocate an array
 * dealloc(void** ptr)
 */
#define dealloc(ptr)                                                                               \
    do {                                                                                           \
        ERRCHK((ptr) != NULL);                                                                     \
        free((ptr));                                                                               \
        (ptr) = NULL;                                                                              \
    } while (0)

/**
 * Resize an array
 * realloc(const size_t count, void** ptr)
 */
#define nrealloc(count, ptr)                                                                       \
    do {                                                                                           \
        (ptr) = realloc((ptr), sizeof((ptr)[0]) * (count));                                        \
        ERRCHK((ptr) != NULL);                                                                     \
    } while (0)

/**
 * Copy an array
 * copy(const size_t count, void* in, void* out)
 */
#define copy(count, in, out)                                                                       \
    do {                                                                                           \
        ERRCHK((in) != NULL);                                                                      \
        ERRCHK((out) != NULL);                                                                     \
        memmove((out), (in), sizeof((in)[0]) * (count));                                           \
    } while (0)

/**
 * Allocate and duplicate an array.
 * dup(const size_t count, const void* in, void* out)
 */
#define dup(count, in, out)                                                                        \
    do {                                                                                           \
        alloc((count), (out));                                                                     \
        copy((count), (in), (out))                                                                 \
    } while (0)

/**
 * Compare two blocks of memory.
 * The arrays must have the same length and element size.
 * cmp(const size_t count, const void* a, const void* b)
 */
#define cmp(count, a, b) (memcmp((a), (b), sizeof((a)[0]) * (count)) == 0 ? true : false)
