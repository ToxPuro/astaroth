#pragma once

#include "func_attributes.h"
template <class T>
static HOST_DEVICE_INLINE const T
min(const T& a, const T& b)
{
  return a < b ? a : b;
}

template <class T>
static HOST_DEVICE_INLINE const T
max(const T& a, const T& b)
{
  return a > b ? a : b;
}

static inline size_t
max(const Volume& a)
{
  const  size_t tmp= a.x > a.y ? a.x : a.y;
  return tmp > a.z ? tmp: a.z;
}

static inline const Volume
max(const Volume& a, const Volume& b)
{
  return (Volume){max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

static inline const Volume
min(const Volume& a, const Volume& b)
{
  return (Volume){min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
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
template <typename T>
static HOST_DEVICE_INLINE Volume 
operator*(const Volume& a, const T& b)
{
  return (Volume){a.x / b, a.y / b, a.z / b};
}

#define to_volume(a) TO_VOLUME(a, __FILE__, __LINE__)

static inline dim3
to_dim3(const Volume v)
{
  return (dim3)
  {
	(unsigned int)v.x,
	(unsigned int)v.y,
	(unsigned int)v.z
  };
}

static HOST_INLINE Volume
operator-(const Volume& a, const Volume& b)
{
	int3 res = 
	{
		(int)a.x - (int)b.x,
		(int)a.y - (int)b.y,
		(int)a.z - (int)b.z
	};

	return to_volume(res);
}
static HOST_INLINE Volume
operator*(const Volume& a, const int& b)
{
	ERRCHK_ALWAYS(b >= 0);
	return
	{
		a.x*(size_t)b,
		a.y*(size_t)b,
		a.z*(size_t)b
	};

}

static HOST_INLINE Volume
operator*(const int& a, const Volume& b)
{
	ERRCHK_ALWAYS(a >= 0);
	return
	{
		b.x*(size_t)a,
		b.y*(size_t)a,
		b.z*(size_t)a
	};

}
