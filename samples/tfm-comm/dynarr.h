#pragma once
#include <stddef.h>

#include "nalloc.h"
#include "type_conversion.h"

/** Contents of the dynamic array
 * Declare new datatypes with typedef dynarr_s(type) name,
 * e.g., typedef dynarr_s(size_t) dynarr_size_t
 */
#define dynarr_s(T)                                                                                \
    struct {                                                                                       \
        size_t length;                                                                             \
        size_t capacity;                                                                           \
        T* data;                                                                                   \
    }

/**
 * Create a new dynamic array
 * dynarr_create(void* arr)
 */
#define dynarr_create(arr)                                                                         \
    do {                                                                                           \
        (arr)->length   = 0;                                                                       \
        (arr)->capacity = 0;                                                                       \
        (arr)->data     = NULL;                                                                    \
    } while (0)

/** Destroy a dynamic array
 * dynarr_destroy(void* arr)
 */
#define dynarr_destroy(arr)                                                                        \
    do {                                                                                           \
        if ((arr)->data != NULL)                                                                   \
            ndealloc((arr)->data);                                                                 \
        (arr)->capacity = 0;                                                                       \
        (arr)->length   = 0;                                                                       \
    } while (0)

/** Append an element to a dynamic array
 * dynarr_append(T elem, void* arr);
 */
#define dynarr_append(elem, arr)                                                                   \
    do {                                                                                           \
        if ((arr)->length == (arr)->capacity)                                                      \
            nrealloc(++(arr)->capacity, (arr)->data);                                              \
        (arr)->data[(arr)->length++] = (elem);                                                     \
    } while (0)

/** Append multiple elements to a dynamic array
 * dynarr_append_multiple(const size_t count, const T* elems, void* arr)
 */
#define dynarr_append_multiple(count, elems, arr)                                                  \
    do {                                                                                           \
        for (size_t __dynarr_i__ = 0; __dynarr_i__ < as_size_t((count)); ++__dynarr_i__)           \
            dynarr_append((elems)[__dynarr_i__], (arr));                                           \
    } while (0)

/** Remove element at index
 * dynarr_remove(const size_t index, void* arr)
 */
#define dynarr_remove(index, arr)                                                                  \
    do {                                                                                           \
        const size_t __dynarr_index__ = as_size_t((index));                                        \
        ERRCHK((__dynarr_index__) < (arr)->length);                                                \
        const size_t __dynarr_count__ = (arr)->length - (__dynarr_index__) - 1;                    \
        if (__dynarr_count__ > 0)                                                                  \
            ncopy(__dynarr_count__, &(arr)->data[(__dynarr_index__) + 1],                          \
                  &(arr)->data[(__dynarr_index__)]);                                               \
        --(arr)->length;                                                                           \
    } while (0)

/** Remove multiple elements from a dynamic array
 * dynarr_remove_multiple(const size_t index, const size_t count, void* arr)
 */
#define dynarr_remove_multiple(index, count, arr)                                                  \
    do {                                                                                           \
        for (size_t __dynarr_i__ = 0; __dynarr_i__ < as_size_t((count)); ++__dynarr_i__)           \
            dynarr_remove((index), (arr));                                                         \
    } while (0)

void test_dynarr(void);
