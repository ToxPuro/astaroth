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
#include "errchk.h"

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
            result += filter[s] * (Scalar)in[i + s - radius];
        }
        out[i] = (AcReal)result;
    }
    return AC_SUCCESS;
}

AcResult
acHostProfileDerz(const AcReal* in, const size_t count, const AcReal grid_spacing, AcReal* out)
{
#if STENCIL_ORDER == 6
    const Scalar grid_spacing_inv = (Scalar)(1 / grid_spacing);
    const Scalar filter[] = {
        (grid_spacing_inv) * (Scalar)(-1) / 60, //
        (grid_spacing_inv) * (Scalar)(3) / 20,  //
        (grid_spacing_inv) * (Scalar)(-3) / 4,  //
        (grid_spacing_inv) * 0,                 //
        (grid_spacing_inv) * (Scalar)(3) / 4,   //
        (grid_spacing_inv) * (Scalar)(-3) / 20, //
        (grid_spacing_inv) * (Scalar)(1) / 60,
    };
#else
    const Scalar filter[] = {};
    (void)grid_spacing;
#endif

    return xcorr(in, count, filter, ARRAY_LENGTH(filter), out);
}

AcResult
acHostProfileDerzz(const AcReal* in, const size_t count, const AcReal grid_spacing, AcReal* out)
{
#if STENCIL_ORDER == 6
    const Scalar grid_spacing_inv_2= (Scalar)(1 / (grid_spacing*grid_spacing));
    const Scalar filter[] = {
        (grid_spacing_inv_2) * (Scalar)(1) / 90,   //
        (grid_spacing_inv_2) * (Scalar)(-3) / 20,  //
        (grid_spacing_inv_2) * (Scalar)(3) / 2,    //
        (grid_spacing_inv_2) * (Scalar)(-49) / 18, //
        (grid_spacing_inv_2) * (Scalar)(3) / 2,    //
        (grid_spacing_inv_2) * (Scalar)(-3) / 20,  //
        (grid_spacing_inv_2) * (Scalar)(1) / 90,
    };
#else
    const Scalar filter[] = {};
    (void)grid_spacing;
#endif

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
                sum += (Scalar)in[idx];
            }
        }
        out[k] = (AcReal)(sum / (dims.nn.x * dims.nn.y));
    }
    return AC_SUCCESS;
}

/** box_size: size of the simulation box in real physical units (usually 2*M_PI)
    nz: number of indices in the global computational domain in the z direction
    offset: offset for the first index off the computational domain
            single GPU: (0 - stencil radius in the z dimension)
            multi-GPU: (pz * local mz - stencil radius) where pz the process
                       index in the z dimension
    profile_count: number of elements in the profile (local mz) */
AcResult
acHostInitProfileToCosineWave(const long double box_size, const size_t nz, const long offset,
                              const AcReal amplitude, const AcReal wavenumber,
                              const size_t profile_count, AcReal* profile)
{
    const long double spacing = box_size / (nz - 1);
    for (size_t i = 0; i < profile_count; ++i) {
        profile[i] = static_cast<AcReal>(static_cast<long double>(amplitude) * cos(static_cast<long double>(wavenumber) * spacing * ((long)i + offset)));
    }
    return AC_SUCCESS;
}

/** See acHostInitProfileToCosineWave */
AcResult
acHostInitProfileToSineWave(const long double box_size, const size_t nz, const long offset,
                            const AcReal amplitude, const AcReal wavenumber,
                            const size_t profile_count, AcReal* profile)
{
    const long double spacing = box_size / (nz - 1);
    for (size_t i = 0; i < profile_count; ++i) {
        profile[i] = static_cast<AcReal>(static_cast<long double>(amplitude) * sin((long double)wavenumber * spacing * ((long)i + offset)));
    }
    return AC_SUCCESS;
}

AcResult
acHostInitProfileToValue(const long double value, const size_t profile_count, AcReal* profile)
{
    for (size_t i = 0; i < profile_count; ++i) {
        profile[i] = static_cast<AcReal>(value);
    }
    return AC_SUCCESS;
}

AcResult
acHostWriteProfileToFile(const char* filepath, const AcReal* profile, const size_t profile_count)
{
    FILE* fp                   = fopen(filepath, "w");
    const size_t count_written = fwrite(profile, sizeof(profile[0]), profile_count, fp);
    ERRCHK_ALWAYS(count_written == count_written);
    fclose(fp);
    return AC_SUCCESS;
}
