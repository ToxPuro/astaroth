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

template <typename T, size_t N> struct VecN {
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Record the number of elements
    __host__ __device__ constexpr size_t count(void) const { return N; }

    // Enable the array[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    // Default constructor
    __host__ __device__ VecN(const T& fill_value = 0)
    {
        for (size_t i = 0; i < count(); ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // VecN<int, 3> a = {1,2,3}
    __host__ __device__ VecN(const std::initializer_list<T>& init_list)
    {
        ERRCHK(init_list.size() == N);
        std::copy(init_list.begin(), init_list.begin() + count(), data);
    }

    // Copy constructor with proper casting
    // VecN<T, N> a(VecN<U, N> b)
    template <typename U> __host__ __device__ VecN(const VecN<U, N>& other)
    {
        for (size_t i = 0; i < count(); ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ VecN(const size_t count_, const T* data_)
    {
        ERRCHK(count_ == count());
        ERRCHK(data_);
        for (size_t i = 0; i < count(); ++i)
            data[i] = data_[i];
    }

    // Common operations
    __host__ __device__ T dot(const VecN<T, N> other)
    {
        T res = 0;
        for (size_t i = 0; i < count(); ++i)
            res += data[i] * other[i];
        return res;
    }

    __host__ VecN<T, N> reversed()
    {
        VecN<T, N> out(count());
        for (size_t i = 0; i < count(); ++i)
            out.data[i] = data[count() - 1 - i];
        return out;
    }
};

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type __host__ __device__
prod(const VecN<T, N> arr)
{
    T result = 1;
    for (size_t i = 0; i < arr.count(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator+(const VecN<T, N>& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator+(const T& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator+(const VecN<T, N>& a, const T& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] + b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator-(const VecN<T, N>& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator-(const T& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator-(const VecN<T, N>& a, const T& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] - b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator*(const VecN<T, N>& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator*(const T& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator*(const VecN<T, N>& a, const T& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] * b;
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator/(const VecN<T, N>& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a[i] / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator/(const T& a, const VecN<T, N>& b)
{
    ERRCHK(a.count() == b.count());
    ERRCHK(b != 0);
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a / b[i];
    }
    return c;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, VecN<T, N>>::type __host__ __device__
operator/(const VecN<T, N>& a, const T& b)
{
    ERRCHK(a.count() == b.count());
    ERRCHK(b != 0);
    VecN<T, N> c(a.count());
    for (size_t i = 0; i < c.count(); ++i)
        c[i] = a[i] / b;
    return c;
}

#include <iostream>

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const VecN<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.count(); ++i)
        os << obj[i] << (i + 1 < obj.count() ? ", " : "}");
    return os;
}

int
test_vecn(void)
{
    int retval = 0;

    // {
    //     VecN<uint64_t, 5> arr = {1, 2, 3, 4, 5};
    //     retval |= WARNCHK(prod(arr) == 120);
    // }

    return retval;
}
