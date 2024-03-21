/*
    Copyright (C) 2024, Johannes Pekkila.

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
#include "astaroth.h"

typedef long double Scalar;
#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

static AcResult
xcorr(const AcReal* in, const size_t count, const Scalar* filter, const size_t filter_width,
      AcReal* out)
{
    const size_t radius = (filter_width - 1) / 2;
    ERRCHK_ALWAYS(2 * radius + 1 == filter_width);
    for (size_t i = radius; i < count - radius; ++i) {
        Scalar result = 0;
        for (size_t s = 0; s < filter_width; ++s) {
            ERRCHK_ALWAYS(i + s - radius < count);
            result += filter[s] * in[i + s - radius];
        }
        out[i] = (AcReal)result;
    }
    return AC_SUCCESS;
}

AcResult
acHostProfileDerz(const AcReal* in, const size_t count, const AcReal grid_spacing, AcReal* out)
{
#if STENCIL_ORDER == 6
    const Scalar filter[] = {
        (1 / grid_spacing) * (Scalar)(-1) / 60, //
        (1 / grid_spacing) * (Scalar)(3) / 20,  //
        (1 / grid_spacing) * (Scalar)(-3) / 4,  //
        (1 / grid_spacing) * 0,                 //
        (1 / grid_spacing) * (Scalar)(3) / 4,   //
        (1 / grid_spacing) * (Scalar)(-3) / 20, //
        (1 / grid_spacing) * (Scalar)(1) / 60,
    };
#endif

    return xcorr(in, count, filter, ARRAY_LENGTH(filter), out);
}

AcResult
acHostProfileDerzz(const AcReal* in, const size_t count, const AcReal grid_spacing, AcReal* out)
{
#if STENCIL_ORDER == 6
    const Scalar filter[] = {
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(1) / 90,   //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(-3) / 20,  //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(3) / 2,    //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(-49) / 18, //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(3) / 2,    //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(-3) / 20,  //
        (1 / (grid_spacing * grid_spacing)) * (Scalar)(1) / 90,
    };
#endif

    return xcorr(in, count, filter, ARRAY_LENGTH(filter), out);

    return xcorr(in, count, filter, ARRAY_LENGTH(filter), out);
}

AcResult
acHostReduceXYAverage(const AcReal* in, const AcMeshDims dims, AcReal* out)
{
    for (size_t k = dims.m0.z; k < as_size_t(dims.m1.z); ++k) {
        Scalar sum = 0;
        for (size_t j = dims.n0.y; j < as_size_t(dims.n1.y); ++j) {
            for (size_t i = dims.n0.x; i < as_size_t(dims.n1.x); ++i) {
                const size_t idx = i + j * dims.m1.x + k * dims.m1.x * dims.m1.y;
                sum += in[idx];
            }
        }
        out[k] = (AcReal)(sum / (dims.nn.x * dims.nn.y));
    }
    return AC_SUCCESS;
}