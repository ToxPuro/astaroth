#pragma once
#include "errchk.h"

#include <string.h>

/**
 * Allocate an array
 * nalloc(const size_t count, void** ptr)
 */
#define nalloc(count, ptr)                                                                         \
    do {                                                                                           \
        (ptr) = malloc(sizeof((ptr)[0]) * (count));                                                \
        ERRCHK((ptr) != NULL);                                                                     \
    } while (0)

/**
 * Allocate an array and set it to zero
 * ncalloc(const size_t count, void** ptr)
 */
#define ncalloc(count, ptr)                                                                        \
    do {                                                                                           \
        (ptr) = calloc((count), sizeof((ptr)[0]));                                                 \
        ERRCHK((ptr) != NULL);                                                                     \
    } while (0)

/**
 * Deallocate an array
 * ndealloc(void** ptr)
 */
#define ndealloc(ptr)                                                                              \
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
 * ncopy(const size_t count, void* in, void* out)
 */
#define ncopy(count, in, out)                                                                      \
    do {                                                                                           \
        ERRCHK((in) != NULL);                                                                      \
        ERRCHK((out) != NULL);                                                                     \
        memmove((out), (in), sizeof((in)[0]) * (count));                                           \
    } while (0)

/**
 * Allocate and duplicate an array.
 * ndup(const size_t count, const void* in, void* out)
 */
#define ndup(count, in, out)                                                                       \
    do {                                                                                           \
        ERRCHK((in) != NULL);                                                                      \
        nalloc((count), (out));                                                                    \
        ncopy((count), (in), (out));                                                               \
    } while (0)

/**
 * Compare two blocks of memory.
 * The arrays must have the same length and element size.
 * ncmp(const size_t count, const void* a, const void* b)
 */
#define ncmp(count, a, b) (memcmp((a), (b), sizeof((a)[0]) * (count)) == 0 ? true : false)
