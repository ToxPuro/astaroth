#pragma once
#include <stddef.h>

#include "nalloc.h"
#include "type_conversion.h"

/** Contents of the dynamic array
 * Declare new datatypes with typedef dynarr_s(type) name,
 * e.g., typedef dynarr_s(size_t) dynarr_size_t
 */
#define dynarr_s(T_)                                                                               \
    struct {                                                                                       \
        size_t length;                                                                             \
        size_t capacity;                                                                           \
        void (*destructor)(T_*);                                                                   \
        T_* data;                                                                                  \
    }

/**
 * Create a new dynamic array
 * dynarr_create(void* arr)
 */
#define dynarr_create(arr_)                                                                        \
    do {                                                                                           \
        (arr_)->length     = 0;                                                                    \
        (arr_)->capacity   = 0;                                                                    \
        (arr_)->destructor = NULL;                                                                 \
        (arr_)->data       = NULL;                                                                 \
    } while (0)

/**
 * Create a new dynamic array with a destructor
 * dynarr_create_with_destructor(void (*destructor)(T*), void* arr)
 */
#define dynarr_create_with_destructor(destructor_, arr_)                                           \
    do {                                                                                           \
        (arr_)->length     = 0;                                                                    \
        (arr_)->capacity   = 0;                                                                    \
        (arr_)->destructor = destructor_;                                                          \
        (arr_)->data       = NULL;                                                                 \
    } while (0)

/** Destroy a dynamic array
 * dynarr_destroy(void* arr)
 */
#define dynarr_destroy(arr_)                                                                       \
    do {                                                                                           \
        if ((arr_)->destructor != NULL)                                                            \
            while ((arr_)->length > 0)                                                             \
                dynarr_remove(0, (arr_));                                                          \
        if ((arr_)->data != NULL)                                                                  \
            ndealloc((arr_)->data);                                                                \
        (arr_)->capacity = 0;                                                                      \
        (arr_)->length   = 0;                                                                      \
    } while (0)

/** Append an element to a dynamic array
 * dynarr_append(T elem, void* arr);
 */
#define dynarr_append(elem_, arr_)                                                                 \
    do {                                                                                           \
        if ((arr_)->length == (arr_)->capacity)                                                    \
            nrealloc(++(arr_)->capacity, (arr_)->data);                                            \
        (arr_)->data[(arr_)->length++] = (elem_);                                                  \
    } while (0)

/** Append multiple elements to a dynamic array
 * dynarr_append_multiple(const size_t count, const T* elems, void* arr)
 */
#define dynarr_append_multiple(count_, elems_, arr_)                                               \
    do {                                                                                           \
        for (size_t __dynarr_i__ = 0; __dynarr_i__ < as_size_t((count_)); ++__dynarr_i__)          \
            dynarr_append((elems_)[__dynarr_i__], (arr_));                                         \
    } while (0)

/** Remove element at index
 * dynarr_remove(const size_t index, void* arr)
 */
#define dynarr_remove(index_, arr_)                                                                \
    do {                                                                                           \
        const size_t __dynarr_index__ = as_size_t((index_));                                       \
        ERRCHK((__dynarr_index__) < (arr_)->length);                                               \
        if ((arr_)->destructor != NULL)                                                            \
            (arr_)->destructor(&(arr_)->data[index_]);                                             \
        const size_t __dynarr_count__ = (arr_)->length - (__dynarr_index__) - 1;                   \
        if (__dynarr_count__ > 0)                                                                  \
            ncopy(__dynarr_count__, &(arr_)->data[(__dynarr_index__) + 1],                         \
                  &(arr_)->data[(__dynarr_index__)]);                                              \
        --(arr_)->length;                                                                          \
    } while (0)

/** Remove multiple elements from a dynamic array
 * dynarr_remove_multiple(const size_t index, const size_t count, void* arr)
 */
#define dynarr_remove_multiple(index_, count_, arr_)                                               \
    do {                                                                                           \
        for (size_t __dynarr_i__ = 0; __dynarr_i__ < as_size_t((count_)); ++__dynarr_i__)          \
            dynarr_remove((index_), (arr_));                                                       \
    } while (0)

void test_dynarr(void);
