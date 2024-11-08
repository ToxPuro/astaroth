#pragma once
#include <cstddef>

#include <iostream>
#include <type_traits>

#if defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#elif defined(__HIP_DEVICE_COMPILE__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#endif

#include "errchk.h"
#include "type_conversion.h"

template <typename T, size_t N> struct StaticArray {
    size_t count;
    T data[N] = {};

    // Record the number of elements
    __host__ __device__ constexpr size_t capacity(void) const { return N; }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor (disabled)
    // __host__ __device__ StaticArray() : count(0), data{} {}

    // Vector-like constructor
    // StaticArray<int, N> a(10, 1)
    __host__ __device__ StaticArray(const size_t in_count, const T& fill_value = 0)
        : count(in_count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        for (size_t i = 0; i < count; ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // StaticArray<int, 3> a = {1,2,3}
    __host__ __device__ explicit StaticArray(const std::initializer_list<T>& init_list)
        : count(init_list.size())
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        std::copy(init_list.begin(), init_list.begin() + count, data);
    }

    // Copy constructor with proper casting
    // StaticArray<T, N> a(StaticArray<U, N> b)
    template <typename U>
    __host__ __device__ explicit StaticArray(const StaticArray<U, N>& other)
        : count(other.count)
    {
        for (size_t i = 0; i < count; ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ StaticArray(const size_t in_count, const T* arr)
        : count(in_count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        ERRCHK(arr);
        for (size_t i = 0; i < count; ++i)
            data[i] = arr[i];
    }

    // Common operations
    template <typename U> T __host__ __device__ dot(const StaticArray<U, N> other) const
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        T res = 0;
        for (size_t i = 0; i < count; ++i)
            res += data[i] * other[i];
        return res;
    }

    __host__ StaticArray<T, N> reversed() const
    {
        StaticArray<T, N> out(count);
        for (size_t i = 0; i < count; ++i)
            out.data[i] = data[count - 1 - i];
        return out;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator+(const StaticArray<T, N>& a,
                                                           const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator+(const T& a, const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(b.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator+(const StaticArray<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator-(const StaticArray<T, N>& a,
                                                           const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator-(const T& a, const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(b.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator-(const StaticArray<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator*(const StaticArray<T, N>& a,
                                                           const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator*(const T& a, const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(b.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator*(const StaticArray<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator/(const StaticArray<T, N>& a,
                                                           const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
        }
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator/(const T& a, const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        StaticArray<T, N> c(b.count);
        for (size_t i = 0; i < c.count; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a / b[i];
        }
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator/(const StaticArray<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] / b;
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator%(const StaticArray<T, N>& a,
                                                           const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator%(const T& a, const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(b.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U>
    friend StaticArray<T, N> __host__ __device__ operator%(const StaticArray<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U>
    friend bool __host__ __device__ operator==(const StaticArray<T, N>& a,
                                               const StaticArray<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same<T, U>::value,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.count == b.count);
        for (size_t i = 0; i < a.count; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    friend StaticArray<T, N> __host__ __device__ operator-(const StaticArray<T, N>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        StaticArray<T, N> c(a.count);
        for (size_t i = 0; i < c.count; ++i)
            c[i] = -a[i];
        return c;
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const StaticArray<T, N>& obj)
    {
        os << "{";
        for (size_t i = 0; i < obj.count; ++i)
            os << obj[i] << (i + 1 < obj.count ? ", " : "}");
        return os;
    }
};

template <typename T, size_t N>
T __host__ __device__
prod(const StaticArray<T, N> arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result = 1;
    for (size_t i = 0; i < arr.count; ++i)
        result *= arr[i];
    return result;
}

// template <typename T, size_t N>
// typename std::enable_if<std::is_arithmetic_v<T>, StaticArray<T, N>>::type __host__
// __device__ operator+(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
// {
//     ERRCHK(a.count == b.count);
//     StaticArray<T, N> c(a.count);
//     for (size_t i = 0; i < c.count; ++i)
//         c[i] = a[i] + b[i];
//     return c;
// }

void test_static_array(void);
