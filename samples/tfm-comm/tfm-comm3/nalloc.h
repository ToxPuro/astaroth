#pragma once
#include "alloc.h"

/*
 * Provides type-specific wrappers for alloc.h module.
 */

#define create_nalloc(T)                                                                           \
    static inline T* nalloc_##T(const size_t count) { return ac_calloc(count, sizeof(T)); }

#define create_ndealloc(T)                                                                         \
    static inline void ndealloc_##T(T** ptr) { ac_free((void**)ptr); }

#define create_nrealloc(T)                                                                         \
    static inline T* nrealloc_##T(const size_t count, T** ptr)                                     \
    {                                                                                              \
        return ac_realloc(count, sizeof(T), (void**)ptr);                                          \
    }

#define create_ncopy(T)                                                                            \
    static inline void ncopy_##T(const size_t count, const T* in, T* out)                          \
    {                                                                                              \
        ac_copy(count, sizeof(T), in, out);                                                        \
    }

#define create_dup(T)                                                                              \
    static inline T* ndup_##T(const size_t count, const T* ptr)                                    \
    {                                                                                              \
        return ac_dup(count, sizeof(T), ptr);                                                      \
    }

#define create_reverse(T)                                                                          \
    static inline void nreverse(const size_t count, T* ptr) { ac_reverse(count, sizeof(T), ptr); }

#define declare_nalloc_type(T)                                                                     \
    create_nalloc(T) create_ndealloc(T) create_nrealloc(T) create_ncopy(T) create_dup(T)           \
        create_reverse(T)

declare_nalloc_type(size_t)

    // #define ndealloc(ptr) _Generic(((*ptr)[0]), size_t: ndealloc_size_t)(ptr)
