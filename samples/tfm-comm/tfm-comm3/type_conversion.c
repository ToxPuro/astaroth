#include "type_conversion.h"

#include <limits.h> // INT_MAX

#include "errchk.h"

#define SIGNED_TO_UNSIGNED(T_FROM, T_TO, MIN_TO, MAX_TO)                                           \
    T_TO T_FROM##_as_##T_TO(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x >= 0);                                                                            \
        ERRCHK((T_TO)x <= MAX_TO);                                                                 \
        return (T_TO)x;                                                                            \
    }

#define UNSIGNED_TO_SIGNED(T_FROM, T_TO, MIN_TO, MAX_TO)                                           \
    T_TO T_FROM##_as_##T_TO(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x <= MAX_TO);                                                                       \
        return (T_TO)x;                                                                            \
    }

#define UNSIGNED_TO_UNSIGNED(T_FROM, T_TO, MIN_TO, MAX_TO)                                         \
    T_TO T_FROM##_as_##T_TO(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x <= MAX_TO);                                                                       \
        return (T_TO)x;                                                                            \
    }

#define SIGNED_TO_SIGNED(T_FROM, T_TO, MIN_TO, MAX_TO)                                             \
    T_TO T_FROM##_as_##T_TO(const T_FROM x)                                                        \
    {                                                                                              \
        ERRCHK(x >= MIN_TO);                                                                       \
        ERRCHK(x <= MAX_TO);                                                                       \
        return (T_TO)x;                                                                            \
    }

#define AS_SELF(T_)                                                                                \
    T_ T_##_as_##T_(const T_ x) { return x; }

AS_SELF(uint64_t)
UNSIGNED_TO_UNSIGNED(size_t, uint64_t, 0, UINT64_MAX)
SIGNED_TO_UNSIGNED(int, uint64_t, 0, UINT64_MAX)
SIGNED_TO_UNSIGNED(int64_t, uint64_t, 0, UINT64_MAX)

AS_SELF(size_t)
UNSIGNED_TO_UNSIGNED(uint64_t, size_t, 0, SIZE_MAX)
SIGNED_TO_UNSIGNED(int, size_t, 0, SIZE_MAX)
SIGNED_TO_UNSIGNED(int64_t, size_t, 0, SIZE_MAX)

AS_SELF(int)
UNSIGNED_TO_UNSIGNED(size_t, int, 0, INT_MAX)
UNSIGNED_TO_SIGNED(uint64_t, int, 0, INT_MAX)
SIGNED_TO_SIGNED(int64_t, int, INT_MIN, INT_MAX)
