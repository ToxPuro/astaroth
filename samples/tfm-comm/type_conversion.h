#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "errchk.h"

template <typename T_target, typename T_source>
constexpr bool
can_convert(T_source value)
{
    static_assert(std::is_integral_v<T_target>, "Conversion enabled only for integral types");
    static_assert(std::is_integral<T_source>::value, "Conversion enabled only for integral types");
    if constexpr (std::is_signed<T_source>::value && std::is_signed<T_target>::value) {
        return value >= std::numeric_limits<T_target>::min() &&
               value <= std::numeric_limits<T_target>::max();
    }
    else if constexpr (std::is_unsigned<T_source>::value && std::is_unsigned<T_target>::value) {
        return value <= std::numeric_limits<T_target>::max();
    }
    else if constexpr (std::is_signed<T_source>::value && std::is_unsigned<T_target>::value) {
        return value >= 0 && static_cast<uintmax_t>(value) <= std::numeric_limits<T_target>::max();
    }
    else if constexpr (std::is_unsigned<T_source>::value && std::is_signed<T_target>::value) {
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
