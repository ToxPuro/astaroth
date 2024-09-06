/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

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
#pragma once

#include <float.h> // DBL/FLT_EPSILON

#include <math.h>
#if AC_USE_HIP
  #include "hip.h"
#else
  #include <vector_types.h> // CUDA vector types
#endif


#if AC_DOUBLE_PRECISION
typedef double AcReal;
#define AC_REAL_MAX (DBL_MAX)
#define AC_REAL_MIN (DBL_MIN)
#define AC_REAL_EPSILON (DBL_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_DOUBLE)
#define AC_REAL_INVALID_VALUE (DBL_MAX)
#else
typedef float AcReal;
#define AC_REAL_MAX (FLT_MAX)
#define AC_REAL_MIN (FLT_MIN)
#define AC_REAL_EPSILON (FLT_EPSILON)
#define AC_REAL_MPI_TYPE (MPI_FLOAT)
#define AC_REAL_INVALID_VALUE (FLT_MAX)
#endif


#define AC_REAL_PI ((AcReal)M_PI)

// convert 3-array into vector
#define TOVEC3(type,arr) ((type){arr[0],arr[1],arr[2]})
#define TOACREAL3(arr) TOVEC3(AcReal3,arr)
#define AcVector AcReal3


typedef enum { AC_SUCCESS = 0, AC_FAILURE = 1 } AcResult;

typedef struct {
  size_t x, y, z;
} Volume;

#include "user_typedefs.h"

#ifdef __cplusplus
#include <initializer_list>
template <typename T, std::size_t N>
class AcArray{
public:
    HOST_DEVICE constexpr T& operator[](const std::size_t index) {
        return arr_[index];
    }

    HOST_DEVICE const constexpr T& operator[](const std::size_t index) const {
        return arr_[index];
    }


    // Additional functions to interact with the internal array
    HOST_DEVICE constexpr std::size_t size() const noexcept {
        return N;
    }

    HOST_DEVICE constexpr T* data() noexcept {
        return arr_;
    }

    HOST_DEVICE constexpr const T* data() const noexcept {
        return arr_;
    }
    HOST_DEVICE constexpr AcArray(std::initializer_list<T> init) : arr_{}{
        std::size_t i = 0;
        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i) {
            arr_[i] = *it;
        }
    }

    HOST_DEVICE constexpr AcArray(void) : arr_{}{}

private:
    T arr_[N];
};
#endif
