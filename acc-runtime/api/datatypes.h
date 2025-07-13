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
#pragma GCC system_header // NOTE: Silences errors originating from HIP and CUDA
                          // headers

#include <float.h> // DBL/FLT_EPSILON

#include <math.h>

#if AC_CPU_BUILD
#else
#if AC_USE_HIP
  #include "hip.h"
  #include <hip/hip_runtime_api.h>
#else
  #include <vector_types.h> // CUDA vector types
  #include <cuda_runtime_api.h> // cuda_assert
#endif

#endif

#include "acreal.h"

// convert 3-array into vector
#define TOVEC3(type,arr) ((type){arr[0],arr[1],arr[2]})
#define TOACREAL3(arr) TOVEC3(AcReal3,arr)
#define AcVector AcReal3


#include "acreal.h"

#if AC_CPU_BUILD
#ifndef INT3_DEFINED
typedef struct
{
	int x,y,z;
} int3;
#define INT3_DEFINED
#endif
#endif


#include "builtin_enums.h"
#include "user_typedefs.h"
#define VOLUME_DEFINED
#define COMPLEX_DEFINED
#define REAL3_DEFINED
#include "host_datatypes.h"



#ifdef __cplusplus
static HOST_DEVICE_INLINE size3_t
operator+(const size3_t& a, const size3_t& b)
{
	return (size3_t)
	{
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	};
}
static HOST_DEVICE_INLINE int3
operator+(const size3_t& a, const int3& b)
{
	return (int3)
	{
		(int)a.x + b.x,
		(int)a.y + b.y,
		(int)a.z + b.z
	};
}
static HOST_DEVICE_INLINE bool
operator==(const size3_t& a, const int3& b)
{
	return
		(int)a.x == b.x &&
		(int)a.y == b.y &&
		(int)a.z == b.z;
}

static HOST_DEVICE_INLINE bool
operator==(const int3& a, const size3_t& b)
{
	return
		a.x == (int)b.x &&
		a.y == (int)b.y &&
		a.z == (int)b.z;
}

#endif
static HOST_INLINE int3
to_int3(const Volume a)
{
	return 
	(int3)
	{
		(int)a.x,
		(int)a.y,
		(int)a.z
	};
}

