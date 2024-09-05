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

static HOST_DEVICE_INLINE AcComplex
operator*(const AcComplex& a, const AcReal& b)
{
  return (AcComplex){a.x* b,a.y * b};
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


static HOST_DEVICE_INLINE int3
operator*(const int3& a, const int3& b)
{
  return (int3){a.x * b.x, a.y * b.y, a.z * b.z};
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
to_volume(const T a)
{
  return (Volume){as_size_t(a.x), as_size_t(a.y), as_size_t(a.z)};
}

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

template <typename T, std::size_t N>
static HOST_DEVICE_INLINE int
size(const AcArray<T,N>& arr)
{
	(void)arr;
	return N;
}

typedef struct AcMatrix {
  //AcReal data[3][3] = {{0}};
  //TP: default initializer will initialize all values to 0.0
  AcArray<AcArray<AcReal,3>,3> data = {};

  HOST_DEVICE AcMatrix() {}

  HOST_DEVICE AcMatrix(const AcReal3 row0, const AcReal3 row1,
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

  HOST_DEVICE constexpr AcReal3 row(const int row) const
  {
    return (AcReal3){data[row][0], data[row][1], data[row][2]};
  }
  HOST_DEVICE AcReal3 col(const int col) const
  {
    return (AcReal3){data[0][col], data[1][col], data[2][col]};
  }

  HOST_DEVICE constexpr AcReal3 operator*(const AcReal3& v) const
  {
    return (AcReal3){
        AC_dot(row(0), v),
        AC_dot(row(1), v),
        AC_dot(row(2), v),
    };
  }

  HOST_DEVICE AcMatrix operator-() const
  {
    return AcMatrix(-row(0), -row(1), -row(2));
  }
  HOST_DEVICE const constexpr AcArray<AcReal,3>& operator[](const size_t index) const {
	  return data[index];
  }
  HOST_DEVICE constexpr AcArray<AcReal,3>& operator[](const size_t index) {
	  return data[index];
  }
} AcMatrix;

static HOST_DEVICE AcMatrix
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

template <size_t N>
class AcMatrixN {
public:
    AcArray<AcArray<AcReal, N>, N> data;
    template<typename... Args,typename = std::enable_if_t<sizeof...(Args) == N>>
    constexpr AcMatrixN(Args&&... args) : data{std::forward<Args>(args)...} {
	     static_assert(sizeof...(Args) == N, "You need to pass N vectors of length N");
    }
    constexpr AcMatrixN(): data{} {}
    constexpr AcArray<AcReal,N> row(const size_t row)  const { return data[row];}
    constexpr AcArray<AcReal,N> col(const size_t col)  const
    { 
	    AcArray<AcReal,N> res{};
	    for(size_t i = 0; i < N; ++i)
		    res[i] = data[i][col];
	    return res;
    }
   constexpr AcArray<AcReal,N> operator*(const AcArray<AcReal,N>& v) const
   {
     AcArray<AcReal,N> res{};
     for(size_t i = 0; i < N; ++i)
	     res[i] = AC_dot(data[i],v);
     return res;
   }
   constexpr AcMatrixN<N> operator-() const
   {
	   AcMatrixN<N> res{};
	   for(size_t i = 0; i < N; ++i)
	           res.data[i] = -data[i];
	   return res;

   }
};
template <const size_t N>
constexpr static AcMatrixN<N>
operator*(const AcReal& v, const AcMatrixN<N>& m)
{
	AcMatrixN<N> res;
	for(size_t i = 0; i < N; ++i)
		res.data[i] = v*m.data[i];
	return res;
}

template <const size_t N>
constexpr static AcMatrixN<N>
operator-(const AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  AcMatrixN<N> res;
  for(size_t i = 0; i < N; ++i)
	  res.data[i] = A.data[i] - B.data[i];
  return res;
}

template <const size_t N>
constexpr static AcMatrixN<N>
operator+(const AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  AcMatrixN<N> res;
  for(size_t i = 0; i < N; ++i)
	  res.data[i] = A.data[i] + B.data[i];
  return res;
}

template <const size_t N>
constexpr static AcMatrixN<N>
operator*(const AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  AcMatrixN<N> res;
  for(size_t i = 0; i < N; ++i)
	  res.data[i] = A.data[i] * B.data[i];
  return res;
}

template <const size_t N>
constexpr static AcMatrixN<N>
operator/(const AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  AcMatrixN<N> res;
  for(size_t i = 0; i < N; ++i)
	  res.data[i] = A.data[i] / B.data[i];
  return res;
}

template <const size_t N>
constexpr static void
operator+=(AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  for(size_t i = 0; i < N; ++i)
	  A.data[i] += B.data[i];
}

template <const size_t N>
constexpr static void
operator-=(AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  for(size_t i = 0; i < N; ++i)
	  A.data[i] -= B.data[i];
}

template <const size_t N>
constexpr static void
operator*=(AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  for(size_t i = 0; i < N; ++i)
	  A.data[i] *= B.data[i];
}

template <const size_t N>
constexpr static void
operator/=(AcMatrixN<N>& A, const AcMatrixN<N>& B)
{
  for(size_t i = 0; i < N; ++i)
	  A.data[i] /= B.data[i];
}



static HOST_DEVICE AcMatrix
operator-(const AcMatrix& A, const AcMatrix& B)
{
  return AcMatrix(A.row(0) - B.row(0), //
                  A.row(1) - B.row(1), //
                  A.row(2) - B.row(2));
}
static HOST_DEVICE_INLINE AcReal
multm2_sym(const AcMatrix& m)
{
//Squared sum of symmetric matix
  AcReal res = m.data[0][0]*m.data[0][0];
  for(int i=1;i<=2;i++){
    res += m.data[i][i]*m.data[i][i];
    for(int j=0;j<=i-1;j++){
      res += 2*m.data[i][j]*m.data[i][j];
    }
  }
  return res;
}
static HOST_DEVICE_INLINE AcReal3
diagonal(const AcMatrix& m)
{
  return (AcReal3){m.data[0][0], m.data[1][1], m.data[2][2]};
}


/*
 * AcTensor
 */
