#pragma once
#include "errchk.h"

#include <string.h>

/**
 * Allocate an array
 * alloc(const size_t count, void** ptr)
 */
#define alloc(count, ptr)                                                                          \
    {                                                                                              \
        (ptr) = malloc(sizeof((ptr)[0]) * (count));                                                \
        ERRCHK((ptr) != NULL);                                                                     \
    }

/**
 * Deallocate an array
 * dealloc(void** ptr)
 */
#define dealloc(ptr)                                                                               \
    {                                                                                              \
        ERRCHK((ptr) != NULL);                                                                     \
        free((ptr));                                                                               \
        (ptr) = NULL;                                                                              \
    }

/**
 * Resize an array
 * realloc(const size_t count, void** ptr)
 */
#define realloc(count, ptr)                                                                        \
    {                                                                                              \
        (ptr) = realloc((ptr), sizeof((ptr)[0]) * (count));                                        \
        ERRCHK((ptr) != NULL);                                                                     \
    }

/**
 * Copy an array
 * copy(const size_t count, void* in, void* out)
 */
#define copy(count, in, out)                                                                       \
    {                                                                                              \
        ERRCHK((in) != NULL);                                                                      \
        ERRCHK((out) != NULL);                                                                     \
        memmove((out), (in), sizeof((in)[0]) * (count));                                           \
    }

/**
 * Allocate and duplicate an array.
 * dup(const size_t count, const void* in, void* out)
 */
#define dup(count, in, out)                                                                        \
    {                                                                                              \
        alloc((count), (out));                                                                     \
        copy((count), (in), (out))                                                                 \
    }
