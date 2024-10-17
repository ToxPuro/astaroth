#pragma once
#include <cstddef>

#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#define __host__
#define __device__
#endif

template <typename T, size_t N> struct StaticArray {
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Record the number of elements
    __host__ __device__ constexpr size_t count(void) const { return N; }

    // Enable the array[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    // Common operations
    __host__ __device__ T dot(const StaticArray<T, N> other)
    {
        T res = 0;
        for (size_t i = 0; i < count(); ++i)
            res += data[i] * other[i];
        return res;
    }
};

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type __host__ __device__
prod(const StaticArray<T, N> arr)
{
    T result = 1;
    for (size_t i = 0; i < arr.count(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const T& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const StaticArray<T, N>& a, const T& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] + b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const T& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const StaticArray<T, N>& a, const T& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] - b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const T& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const StaticArray<T, N>& a, const T& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] * b;
    return c;
}

#include <iostream>

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const StaticArray<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.count(); ++i)
        os << obj[i] << (i + 1 < obj.count() ? ", " : "}");
    return os;
}
