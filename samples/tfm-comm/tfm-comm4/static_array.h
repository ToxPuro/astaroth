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

#include "errchk.h"
#include "type_conversion.h"

template <typename T, size_t N> struct StaticArray {
    size_t count;
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor (disabled)
    // __host__ __device__ StaticArray() : count(0), data{} {}

    // Vector-like constructor
    // StaticArray<int, N> a(10, 1)
    __host__ __device__ StaticArray(const size_t count, const T& fill_value = 0) : count(count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        for (size_t i = 0; i < count; ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // StaticArray<int, 3> a = {1,2,3}
    __host__ __device__ StaticArray(const std::initializer_list<T>& init_list)
        : count(init_list.size())
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        std::copy(init_list.begin(), init_list.begin() + count, data);
    }

    // Copy constructor with proper casting
    // StaticArray<T, N> a(StaticArray<U, N> b)
    template <typename U>
    __host__ __device__ StaticArray(const StaticArray<U, N>& other) : count(other.count)
    {
        for (size_t i = 0; i < count; ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ StaticArray(const size_t count, const T* arr) : count(count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        ERRCHK(arr);
        for (size_t i = 0; i < count; ++i)
            data[i] = arr[i];
    }

    // Record the number of elements
    __host__ __device__ constexpr size_t capacity(void) const { return N; }

    // Enable the array[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    // Common operations
    __host__ __device__ T dot(const StaticArray<T, N> other)
    {
        T res = 0;
        for (size_t i = 0; i < count; ++i)
            res += data[i] * other[i];
        return res;
    }

    __host__ StaticArray<T, N> reversed()
    {
        StaticArray<T, N> out(count);
        for (size_t i = 0; i < count; ++i)
            out.data[i] = data[count - 1 - i];
        return out;
    }
};

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type __host__ __device__
prod(const StaticArray<T, N> arr)
{
    T result = 1;
    for (size_t i = 0; i < arr.count; ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const T& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const StaticArray<T, N>& a, const T& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] + b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const T& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator-(const StaticArray<T, N>& a, const T& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] - b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const T& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator*(const StaticArray<T, N>& a, const T& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] * b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a[i] / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const T& a, const StaticArray<T, N>& b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(b != 0);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator/(const StaticArray<T, N>& a, const T& b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(b != 0);
    StaticArray<T, N> c(a.count);
    for (size_t i = 0; i < c.count; ++i)
        c[i] = a[i] / b;
    return c;
}

#include <iostream>

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const StaticArray<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.count; ++i)
        os << obj[i] << (i + 1 < obj.count ? ", " : "}");
    return os;
}

int
test_static_array(void)
{
    int retval = 0;

    // {
    //     StaticArray<uint64_t, 5> arr = {1, 2, 3, 4, 5};
    //     retval |= WARNCHK(prod(arr) == 120);
    // }

    return retval;
}
