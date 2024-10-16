#include "type_conversion.h"

#include <limits.h> // INT_MAX

#include "errchk.h"

#define SIGNED_AS_UNSIGNED(T_FROM, T_AS, MIN_AS, MAX_AS)                                           \
    T_AS T_FROM##_as_##T_AS(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x >= 0);                                                                            \
        ERRCHK((uintmax_t)x <= MAX_AS);                                                            \
        return (T_AS)x;                                                                            \
    }

#define UNSIGNED_AS(T_FROM, T_AS, MAX_AS)                                                          \
    T_AS T_FROM##_as_##T_AS(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x <= MAX_AS);                                                                       \
        return (T_AS)x;                                                                            \
    }

#define SIGNED_AS_SIGNED(T_FROM, T_AS, MIN_AS, MAX_AS)                                             \
    T_AS T_FROM##_as_##T_AS(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x >= MIN_AS);                                                                       \
        ERRCHK(x <= MAX_AS);                                                                       \
        return (T_AS)x;                                                                            \
    }

#define AS_SELF(T_)                                                                                \
    T_ T_##_as_##T_(const T_ x) { return x; }

AS_SELF(uint64_t)
UNSIGNED_AS(size_t, uint64_t, UINT64_MAX)
SIGNED_AS_UNSIGNED(int, uint64_t, 0, UINT64_MAX)
SIGNED_AS_UNSIGNED(int64_t, uint64_t, 0, UINT64_MAX)

AS_SELF(size_t)
UNSIGNED_AS(uint64_t, size_t, SIZE_MAX)
SIGNED_AS_UNSIGNED(int, size_t, 0, SIZE_MAX)
SIGNED_AS_UNSIGNED(int64_t, size_t, 0, SIZE_MAX)

AS_SELF(int)
UNSIGNED_AS(size_t, int, INT_MAX)
UNSIGNED_AS(uint64_t, int, INT_MAX)
SIGNED_AS_SIGNED(int64_t, int, INT_MIN, INT_MAX)
