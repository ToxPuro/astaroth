/*
    Copyright (C) 2014-2018, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#pragma once
#include <math.h>   // isnan, isinf
#include <stdlib.h> // rand

template <class T>
static inline const T
max(const T& a, const T& b)
{
    return a > b ? a : b;
}

template <class T>
static inline const T
min(const T& a, const T& b)
{
    return a < b ? a : b;
}

static inline const int3
max(const int3& a, const int3& b)
{
    return (int3){max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

static inline const int3
min(const int3& a, const int3& b)
{
    return (int3){min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

template <class T>
static inline const T
sum(const T& a, const T& b)
{
    return a + b;
}

template <class T>
static inline const T
is_valid(const T& val)
{
    if (isnan(val) || isinf(val))
        return false;
    else
        return true;
}

template <class T>
static inline const T
clamp(const T& val, const T& min, const T& max)
{
    return val < min ? min : val > max ? max : val;
}

static inline AcReal
randr()
{
    return AcReal(rand()) / AcReal(RAND_MAX);
}

static inline int3
operator+(const int3& a, const int3& b)
{
    return (int3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline int3
operator-(const int3& a, const int3& b)
{
    return (int3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline bool
is_power_of_two(const unsigned val)
{
    return val && !(val & (val - 1));
}
