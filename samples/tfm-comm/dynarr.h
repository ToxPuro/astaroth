#pragma once

#include "errchk.h"
#include "nalloc.h"
#include "type_conversion.h"

#define dynarr(T)                                                                                  \
    typedef struct {                                                                               \
        size_t length;                                                                             \
        size_t capacity;                                                                           \
        T* data;                                                                                   \
    } dynarr_##T;

#define dynarr_create(T)                                                                           \
    static inline dynarr_##T dynarr_create_##T(const size_t capacity)                              \
    {                                                                                              \
        dynarr_##T arr = (dynarr_##T){                                                             \
            .length   = 0,                                                                         \
            .capacity = capacity,                                                                  \
            .data     = NULL,                                                                      \
        };                                                                                         \
        nalloc(capacity, arr.data);                                                                \
        return arr;                                                                                \
    }

#define dynarr_destroy(T)                                                                          \
    static inline void dynarr_destroy_##T(dynarr_##T* array)                                       \
    {                                                                                              \
        ndealloc(array->data);                                                                     \
        array->length   = 0;                                                                       \
        array->capacity = 0;                                                                       \
    }

#define dynarr_append(T)                                                                           \
    static inline void dynarr_append_##T(const T element, dynarr_##T* array)                       \
    {                                                                                              \
        if (array->length == array->capacity) {                                                    \
            array->capacity += 128;                                                                \
            nrealloc(array->capacity, array->data);                                                \
        }                                                                                          \
        array->data[array->length] = element;                                                      \
        ++array->length;                                                                           \
    }

#define dynarr_append_multiple(T)                                                                  \
    static inline void dynarr_append_multiple_##T(const size_t count, const T* elements,           \
                                                  dynarr_##T* array)                               \
    {                                                                                              \
        for (size_t i = 0; i < count; ++i)                                                         \
            dynarr_append_##T(elements[i], array);                                                 \
    }

#define dynarr_remove(T)                                                                           \
    static inline void dynarr_remove_##T(const size_t index, dynarr_##T* array)                    \
    {                                                                                              \
        ERRCHK(index < array->length);                                                             \
        const size_t count = array->length - index - 1;                                            \
        if (count > 0)                                                                             \
            ncopy(count, &array->data[index + 1], &array->data[index]);                            \
        --array->length;                                                                           \
    }

#define dynarr_get(T)                                                                              \
    static inline T dynarr_get_##T(const dynarr_##T array, const size_t index)                     \
    {                                                                                              \
        ERRCHK(index < array.length);                                                              \
        return array.data[index];                                                                  \
    }

#define dynarr_test(T)                                                                             \
    static inline void dynarr_test_##T(void)                                                       \
    {                                                                                              \
        {                                                                                          \
            const size_t capacity = 10;                                                            \
            dynarr_##T arr        = dynarr_create_##T(capacity);                                   \
            dynarr_append_##T(as_##T(1), &arr);                                                    \
            dynarr_append_##T(as_##T(2), &arr);                                                    \
            dynarr_append_##T(as_##T(3), &arr);                                                    \
            ERRCHK(dynarr_get_##T(arr, 1) == 2);                                                   \
            dynarr_remove_##T(1, &arr);                                                            \
            ERRCHK(dynarr_get_##T(arr, 0) == 1);                                                   \
            ERRCHK(dynarr_get_##T(arr, 1) == 3);                                                   \
            T* elems;                                                                              \
            const size_t count = 2;                                                                \
            nalloc(2, elems);                                                                      \
            for (size_t i = 0; i < count; ++i)                                                     \
                elems[i] = as_##T(10 + i);                                                         \
            dynarr_append_multiple_##T(count, elems, &arr);                                        \
            ERRCHK(dynarr_get_##T(arr, 0) == 1);                                                   \
            ERRCHK(dynarr_get_##T(arr, 1) == 3);                                                   \
            ERRCHK(dynarr_get_##T(arr, 2) == 10);                                                  \
            ERRCHK(dynarr_get_##T(arr, 3) == 11);                                                  \
            ndealloc(elems);                                                                       \
            dynarr_destroy_##T(&arr);                                                              \
        }                                                                                          \
    }

// clang-format off
#define dynarr_define(T)\
    dynarr(T)\
    dynarr_create(T)\
    dynarr_destroy(T)\
    dynarr_append(T)\
    dynarr_append_multiple(T)\
    dynarr_remove(T)\
    dynarr_get(T)\
    dynarr_test(T)

dynarr_define(size_t)
// dynarr_define(int64_t)
// dynarr_define(int)
    // clang-format on

    static inline void test_dynarr(void)
{
    dynarr_test_size_t();
    // dynarr_test_int64_t();
    // dynarr_test_int();
}

#undef dynarr_define
#undef dynarr_test
#undef dynarr_get
#undef dynarr_remove
#undef dynarr_append_multiple
#undef dynarr_append
#undef dynarr_destroy
#undef dynarr_create
#undef dynarr
#undef define_dynarr
