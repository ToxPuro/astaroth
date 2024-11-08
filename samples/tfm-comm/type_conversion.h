#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "errchk.h"

template <typename T_target, typename T_source>
bool
can_convert(T_source value)
{
    static_assert(std::is_integral_v<T_target>, "Conversion enabled only for integral types");
    static_assert(std::is_integral_v<T_source>, "Conversion enabled only for integral types");
    if (std::is_signed_v<T_source> && std::is_signed_v<T_target>) {
        return value >= std::numeric_limits<T_target>::min() &&
               value <= std::numeric_limits<T_target>::max();
    }
    else if (std::is_unsigned_v<T_source> && std::is_unsigned_v<T_target>) {
        return value <= std::numeric_limits<T_target>::max();
    }
    else if (std::is_signed_v<T_source> && std::is_unsigned_v<T_target>) {
        return value >= 0 && static_cast<uintmax_t>(value) <= std::numeric_limits<T_target>::max();
    }
    else if (std::is_unsigned_v<T_source> && std::is_signed_v<T_target>) {
        return value <= static_cast<uintmax_t>(std::numeric_limits<T_target>::max());
    }
}

template <typename T_target, typename T_source>
T_target
as(T_source value)
{
    ERRCHK((can_convert<T_target, T_source>(value)));
    return static_cast<T_target>(value);
}

void test_type_conversion(void);
