/*/
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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#pragma once
#include <stdint.h> // uint64_t
#include <stdlib.h> // rand
#include <math.h>   // isnan, isinf

#include "datatypes.h"

#if AC_DOUBLE_PRECISION != 1
#ifndef __cplusplus
#define exp(x) expf(x)
#define sin(x) sinf(x)
#define cos(x) cosf(x)
#define sqrt(x) sqrtf(x)
#define fabs(x) fabsf(x)
#endif
#endif

#include "func_attributes.h"

#define ENABLE_COMPLEX_DATATYPE (1)
#if ENABLE_COMPLEX_DATATYPE

#if AC_USE_HIP
#else
static HOST_DEVICE_INLINE void
operator -=(int3& lhs, const int3& rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	lhs.z -= rhs.z;
}

static HOST_DEVICE_INLINE int3
operator*(const int3& a, const int3& b)
{
  return (int3){a.x * b.x, a.y * b.y, a.z * b.z};
}
#endif
static HOST_DEVICE_INLINE AcComplex
exp(const AcComplex& val)
{
  return (AcComplex){exp(val.x) * cos(val.y), exp(val.x) * sin(val.y)};
}

static HOST_DEVICE_INLINE AcComplex
operator*(const AcComplex& a, const AcComplex& b)
{
  return (AcComplex){a.x*b.x - a.y*b.y,a.x*b.y + b.x*a.y};
}

#endif // ENABLE_COMPLEX_DATATYPE

typedef struct uint3_64 {
  uint64_t x, y, z;
  uint3_64(const int3& a):
    x(static_cast<uint64_t>(a.x)), y(static_cast<uint64_t>(a.y)), z(static_cast<uint64_t>(a.z)) {}
  uint3_64(const uint64_t _x, const uint64_t _y, const uint64_t _z):
    x(_x), y(_y), z(_z) {}
  uint3_64(): 
    x(), y(), z() {}
  explicit inline constexpr operator int3() const
  {
    return (int3){(int)x, (int)y, (int)z};
  }
} uint3_64;

template <class T>
static HOST_DEVICE_INLINE const T
val(const T& a)
{
  return a;
}

template <class T>
static HOST_DEVICE_INLINE const T
sum(const T& a, const T& b)
{
  return a + b;
}

template <class T>
static HOST_DEVICE_INLINE const T
max(const T& a, const T& b)
{
  return a > b ? a : b;
}

template <class T>
static HOST_DEVICE_INLINE const T
max(const T& a, const T& b, const T& c)
{
	const auto tmp = a > b ? a : b;
	return (tmp > c) ? tmp : c;
}

template <class T>
static HOST_DEVICE_INLINE const T
min(const T& a, const T& b)
{
  return a < b ? a : b;
}

template <class T>
static HOST_DEVICE_INLINE const T
min(const T& a, const T& b, const T& c)
{
	const auto tmp = a < b ? a : b;
	return (tmp < c) ? tmp : c;
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
clamp(const T& val, const T& min, const T& max)
{
  return val < min ? min : val > max ? max : val;
}

static inline uint64_t
mod(const int a, const int b)
{
  const int r = a % b;
  return r < 0 ? as_size_t(r + b) : as_size_t(r);
}

static inline AcReal
randr()
{
  return AcReal(rand()) / AcReal(RAND_MAX);
}

static inline bool
is_power_of_two(const unsigned val)
{
  return val && !(val & (val - 1));
}

/*
 * INT3
 */
static HOST_DEVICE_INLINE int3
operator+(const int3& a, const int3& b)
{
  return (int3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a)
{
  return (int3){-a.x, -a.y, -a.z};
}

static HOST_DEVICE_INLINE AcReal3
operator*(const int3& a, const AcReal3& b)
{
  return (AcReal3){a.x * b.x, a.y * b.y, a.z * b.z};
}

static HOST_DEVICE_INLINE AcReal3
operator*(const AcReal3& a, const int3& b)
{
  return (AcReal3){a.x * b.x, a.y * b.y, a.z * b.z};
}

static HOST_DEVICE_INLINE int3
operator*(const int& a, const int3& b)
{
  return (int3){a * b.x, a * b.y, a * b.z};
}

static HOST_DEVICE_INLINE int3
operator+(const int3& a, const int b)
{
    return (int3){a.x + b, a.y + b, a.z + b};
}

static HOST_DEVICE_INLINE int3
operator+(const int a, const int3& b)
{
    return (int3){a + b.x, a + b.y, a + b.z};
}

static HOST_DEVICE_INLINE int3
operator-(const int3& a, const int3& b)
{
    return (int3){a.x - b.x, a.y - b.y, a.z - b.z};
}


static HOST_DEVICE_INLINE int3
operator-(const int a, const int3& b)
{
    return (int3){a - b.x, a - b.y, a - b.z};
}

static HOST_DEVICE_INLINE bool
operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

static HOST_DEVICE_INLINE bool
operator!=(const int3& a, const int3& b)
{
  return !(a == b);
}

static HOST_DEVICE_INLINE bool
operator>=(const int3& a, const int3& b)
{
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

static HOST_DEVICE_INLINE bool
operator<=(const int3& a, const int3& b)
{
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

/*
 * UINT3_64
 */
static HOST_INLINE uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static HOST_INLINE uint3_64
operator-(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x - b.x, a.y - b.y, a.z - b.z};
}

static HOST_INLINE uint3_64
operator*(const uint3_64& a, const uint3_64& b)
{
  return (uint3_64){a.x * b.x, a.y * b.y, a.z * b.z};
}

static inline uint3_64
operator*(const int& a, const uint3_64& b)
{
  return (uint3_64){as_size_t(a) * b.x, as_size_t(a) * b.y, as_size_t(a) * b.z};
}

static HOST_DEVICE_INLINE bool
operator==(const uint3_64& a, const uint3_64& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

/*
 * lume
 */
template <class T>
static Volume
TO_VOLUME(const T a, const char* file, const int line)
{
  INDIRECT_ERRCHK_ALWAYS(a.x >= 0,file,line);
  INDIRECT_ERRCHK_ALWAYS(a.y >= 0,file,line);
  INDIRECT_ERRCHK_ALWAYS(a.z >= 0,file,line);
  return (Volume){as_size_t(a.x), as_size_t(a.y), as_size_t(a.z)};
}
static Volume UNUSED
TO_VOLUME(const dim3 a, const char* , const int)
{
  return (Volume){a.x, a.y, a.z};
}

static Volume UNUSED
TO_VOLUME(const size3_t a, const char* , const int)
{
  return (Volume){a.x, a.y, a.z};
}

#define to_volume(a) TO_VOLUME(a, __FILE__, __LINE__)

static inline dim3
to_dim3(const Volume v)
{
  return dim3(v.x, v.y, v.z);
}
template <class T>
static int
volume_size(const T a)
{
  return a.x*a.y*a.z; 
}

/*
 * AcBool3
 */
static HOST_DEVICE bool
any(const AcBool3& a){
    return a.x || a.y || a.z;
}
static HOST_DEVICE bool
all(const AcBool3& a){
    return a.x && a.y && a.z;
}

/*
 * AcReal
 */
static HOST_DEVICE_INLINE bool
is_valid(const AcReal a)
{
  return !isnan(a) && !isinf(a);
}


static HOST_DEVICE_INLINE AcReal
AC_dot(const AcReal2& a, const AcReal2& b)
{
  return a.x * b.x + a.y * b.y;
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal2& a)
{
  return is_valid(a.x) && is_valid(a.y);
}

/*
 * AcReal3
 */


//static HOST_DEVICE_INLINE AcBool3 
//operator!=(const AcReal3& a, const AcReal b)
//{
//  return (AcBool3){
//      a.x != b,
//      a.y != b,
//      a.z != b
//  };
//}
//
//
//static HOST_DEVICE_INLINE AcBool3 
//operator==(const AcReal3& a, const AcReal b)
//{
//  return (AcBool3){
//      a.x == b,
//      a.y == b,
//      a.z == b
//  };
//}



static HOST_DEVICE_INLINE AcReal
sum(const AcReal3& a)
{
    return a.x+a.y+a.z;
}

static HOST_DEVICE_INLINE AcReal
AC_dot(const AcReal3& a, const AcReal3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static HOST_DEVICE_INLINE AcReal
AC_dot(const int3& a, const AcReal3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static HOST_DEVICE_INLINE AcReal
AC_dot(const AcReal3& a, const int3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static HOST_DEVICE_INLINE AcReal3
AC_cross(const AcReal3& a, const AcReal3& b)
{
  return
  {
	a.y * b.z - a.z * b.y,
	a.z * b.x - a.x * b.z,
	a.x * b.y - a.y * b.x,  
  };
}

static HOST_DEVICE_INLINE bool
is_valid(const AcReal3& a)
{
  return is_valid(a.x) && is_valid(a.y) && is_valid(a.z);
}

/*
 * AcMatrix
 */


typedef struct AcMatrix {
  AcReal data[3][3]{};
  HOST_DEVICE_INLINE AcMatrix() {}

  HOST_DEVICE_INLINE AcMatrix(const AcReal3 row0, const AcReal3 row1,
                       const AcReal3 row2)
  {
    data[0][0] = row0.x;
    data[0][1] = row0.y;
    data[0][2] = row0.z;

    data[1][0] = row1.x;
    data[1][1] = row1.y;
    data[1][2] = row1.z;

    data[2][0] = row2.x;
    data[2][1] = row2.y;
    data[2][2] = row2.z;
  }

  HOST_DEVICE_INLINE AcReal3 row(const int row) const
  {
    return (AcReal3){data[row][0], data[row][1], data[row][2]};
  }
  HOST_DEVICE_INLINE AcReal3 col(const int col) const
  {
    return (AcReal3){data[0][col], data[1][col], data[2][col]};
  }

  HOST_DEVICE_INLINE AcReal3 operator*(const AcReal3& v) const
  {
    return (AcReal3){
        AC_dot(row(0), v),
        AC_dot(row(1), v),
        AC_dot(row(2), v),
    };
  }

  HOST_DEVICE_INLINE AcMatrix operator-() const
  {
    return AcMatrix(-row(0), -row(1), -row(2));
  }
} AcMatrix;

static HOST_DEVICE_INLINE AcMatrix
operator*(const AcReal& v, const AcMatrix& m)
{
  AcMatrix out;

  out.data[0][0] = v * m.data[0][0];
  out.data[0][1] = v * m.data[0][1];
  out.data[0][2] = v * m.data[0][2];

  out.data[1][0] = v * m.data[1][0];
  out.data[1][1] = v * m.data[1][1];
  out.data[1][2] = v * m.data[1][2];

  out.data[2][0] = v * m.data[2][0];
  out.data[2][1] = v * m.data[2][1];
  out.data[2][2] = v * m.data[2][2];

  return out;
}


/**
#define GEN_STD_ARRAY_OPERATOR(OPERATOR)  \
template <typename T, const size_t N, typename F> \
static constexpr  void \
operator OPERATOR##=(AcArray<T, N>& lhs, const F& rhs) {\
    for (std::size_t i = 0; i < N; ++i) \
        lhs[i] OPERATOR##= rhs;\
}\
template <typename T, const size_t N, typename F> \
static constexpr  AcArray<T, N>  \
operator OPERATOR (const AcArray<T, N>& lhs, const F& rhs) {\
    AcArray<T, N> result = {}; \
    for (std::size_t i = 0; i < N; ++i) {\
        result[i] = lhs[i] OPERATOR rhs;\
    }\
    return result; \
}\
template <typename T, const size_t N, typename F> \
static constexpr  AcArray<T, N>  \
operator OPERATOR (const F& rhs,const AcArray<T, N>& lhs) {\
    AcArray<T, N> result = {}; \
    for (std::size_t i = 0; i < N; ++i) {\
        result[i] = lhs[i] OPERATOR rhs;\
    }\
    return result; \
}\
template <typename T, const size_t N> \
static constexpr  AcArray<T, N>  \
operator OPERATOR (const AcArray<T, N>& lhs, const AcArray<T, N>& rhs) {\
    AcArray<T, N> result = {}; \
    for (std::size_t i = 0; i < N; ++i) {\
        result[i] = lhs[i] OPERATOR rhs[i];\
    }\
    return result; \
}\
template <typename T, const size_t N> \
static constexpr  void \
operator OPERATOR##=(AcArray<T, N>& lhs, const AcArray<T, N>& rhs) {\
    for (std::size_t i = 0; i < N; ++i) \
        lhs[i] OPERATOR##= rhs[i];\
}\



GEN_STD_ARRAY_OPERATOR(*)
GEN_STD_ARRAY_OPERATOR(/)
GEN_STD_ARRAY_OPERATOR(+)
GEN_STD_ARRAY_OPERATOR(-)

template <typename T, std::size_t N>
static HOST_DEVICE AcArray<T, N>  
operator -(const AcArray<T, N>& lhs) {
    AcArray<T, N> result; 
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = -lhs[i];
    }
    return result;
}

template <typename T, std::size_t N>
static HOST_DEVICE AcArray<T, N>  
operator +(const AcArray<T, N>& lhs) {
    AcArray<T, N> result = lhs; 
    return result;
}

template <typename T, std::size_t N>
static HOST_DEVICE_INLINE T
AC_dot(const AcArray<T,N>& a, const AcArray<T,N>& b)
{
        T res = 0;
        for(size_t i = 0; i < N; ++i)
                res += a[i]*b[i];
        return res;
}
**/
