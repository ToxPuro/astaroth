#pragma once
#include <cstddef>

#include <iostream>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#define __host__
#define __device__
#endif

#include "errchk.h"
#include "type_conversion.h"

template <typename T, size_t N> struct StaticArray {
    T data[N];

    // Record the number of elements
    __host__ __device__ constexpr size_t count() const { return N; }

    // Enable the array[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor with an optional fill_value parameter
    __host__ __device__ StaticArray(const T& fill_value = 0)
    {
        for (size_t i = 0; i < N; ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // StaticArray<int, 3> a = {1,2,3}
    __host__ __device__ StaticArray(const std::initializer_list<T>& init_list)
    {
        ERRCHK(init_list.size() == N);
        std::copy(init_list.begin(), init_list.end(), data);
    }

    // Copy constructor with proper casting
    // StaticArray<T, N> a(StaticArray<U, N> b)
    template <typename U> __host__ __device__ StaticArray(const StaticArray<U, N>& other)
    {
        for (size_t i = 0; i < N; ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ StaticArray(const size_t count, const T* arr)
    {
        ERRCHK(count == N);
        ERRCHK(arr);
        for (size_t i = 0; i < count; ++i)
            data[i] = arr[i];
    }

    // Common operations
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type __host__ __device__
    dot(const StaticArray<T, N> other)
    {
        T res = 0;
        for (size_t i = 0; i < N; ++i)
            res += data[i] * other.data[i];
        return res;
    }

    __host__ StaticArray<T, N> reversed()
    {
        StaticArray<T, N> out;
        for (size_t i = 0; i < N; ++i)
            out.data[i] = data[N - 1 - i];
        return out;
    }
};

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type __host__ __device__
prod(const StaticArray<T, N> arr)
{
    T result = 1;
    for (size_t i = 0; i < N; ++i)
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

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a[i] / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const T& a, const StaticArray<T, N>& b)
{
    ERRCHK(b != 0);
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const StaticArray<T, N>& a, const T& b)
{
    ERRCHK(b != 0);
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] / b;
    return c;
}

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const StaticArray<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.count(); ++i)
        os << obj.data[i] << (i + 1 < obj.count() ? ", " : "}");
    return os;
}
