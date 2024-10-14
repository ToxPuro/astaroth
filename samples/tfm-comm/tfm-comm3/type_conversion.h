#pragma once
#include <stddef.h>
#include <stdint.h>

#define AS_OTHER_DECL(T_FROM, T_TO) T_TO T_FROM##_as_##T_TO(const T_FROM x)

#define AS_SELF_DECL(T_) T_ T_##_as_##T_(const T_ x)

AS_SELF_DECL(uint64_t);
AS_OTHER_DECL(size_t, uint64_t);
AS_OTHER_DECL(int, uint64_t);
AS_OTHER_DECL(int64_t, uint64_t);

AS_SELF_DECL(size_t);
AS_OTHER_DECL(uint64_t, size_t);
AS_OTHER_DECL(int, size_t);
AS_OTHER_DECL(int64_t, size_t);

AS_SELF_DECL(int);
AS_OTHER_DECL(size_t, int);
AS_OTHER_DECL(uint64_t, int);
AS_OTHER_DECL(int64_t, int);

#define as_uint64_t(x)                                                                             \
    _Generic((x),                                                                                  \
        uint64_t: uint64_t_as_uint64_t,                                                            \
        size_t: size_t_as_uint64_t,                                                                \
        int: int_as_uint64_t,                                                                      \
        int64_t: int64_t_as_uint64_t)(x)

#define as_size_t(x)                                                                               \
    _Generic((x),                                                                                  \
        uint64_t: uint64_t_as_size_t,                                                              \
        size_t: size_t_as_size_t,                                                                  \
        int: int_as_size_t,                                                                        \
        int64_t: int64_t_as_size_t)(x)

#define as_int(x)                                                                                  \
    _Generic((x),                                                                                  \
        uint64_t: uint64_t_as_int,                                                                 \
        size_t: size_t_as_int,                                                                     \
        int: int_as_int,                                                                           \
        int64_t: int64_t_as_int)(x)

#undef AS_OTHER_DECL
#undef AS_SELF_DECL
