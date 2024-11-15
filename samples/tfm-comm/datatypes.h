#pragma once
#include <iostream>

#include "type_conversion.h"

#if defined(DEVICE_ENABLED)
// Common GPU
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
namespace ac {
template <typename T> using host_vector        = thrust::host_vector<T>;
template <typename T> using pinned_host_vector = thrust::host_vector<T>; // TODO
template <typename T> using device_vector      = thrust::device_vector<T>;
using thrust::copy;
using thrust::multiplies;
using thrust::reduce;
} // namespace ac
#if defined(CUDA_ENABLED)
// CUDA-specific
#include <cuda/std/array>
namespace ac {
template <typename T, size_t N> using array = cuda::std::array<T, N>;
}
#elif defined(HIP_ENABLED)
// HIP-specific
namespace ac {
template <typename T, size_t N> using array = std::array<T, N>;
}
#endif
#else
#include <vector>
namespace ac {
template <typename T> using host_vector        = std::vector<T>;
template <typename T> using pinned_host_vector = std::vector<T>;
template <typename T> using device_vector      = std::vector<T>;
template <typename T, size_t N> using array    = std::array<T, N>;
using std::copy;
using std::multiplies;
using std::reduce;
} // namespace ac
#define __host__
#define __device__
#endif

constexpr size_t NDIMS = 3;

template <size_t N> using Index     = ac::array<uint64_t, N>;
template <size_t N> using Shape     = ac::array<uint64_t, N>;
template <size_t N> using Direction = ac::array<int64_t, N>;
template <size_t N> using MPIIndex  = ac::array<int, N>;
template <size_t N> using MPIShape  = ac::array<int, N>;
using AcReal                        = double;

template <typename T = uint64_t>
[[nodiscard]] auto
ones()
{
    ac::array<T, NDIMS> arr{};
    std::fill_n(arr.begin(), NDIMS, as<T>(1));
    return arr;
};

template <typename T = uint64_t>
[[nodiscard]] auto
fill(const T& fill_value)
{
    ac::array<T, NDIMS> arr{};
    std::fill_n(arr.begin(), NDIMS, fill_value);
    return arr;
};

// template <typename T, size_t N>
// T __host__ __device__
// dot(const ac::array<T, N> other)
// {
//     static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
//     static_assert(std::is_same_v<T, U>,
//                   "Operator not enabled for parameters of different types. Perform an "
//                   "explicit cast such that both operands are of the same type");
//     T res = 0;
//     for (size_t i = 0; i < count; ++i)
//         res += data[i] * other[i];
//     return res;
// }

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator+(const ac::array<T, N>& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(a.size() == b.size());
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator+(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator+(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] + b;
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const ac::array<T, N>& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(a.size() == b.size());
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] - b;
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator*(const ac::array<T, N>& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(a.size() == b.size());
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator*(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator*(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] * b;
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator/(const ac::array<T, N>& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(a.size() == b.size());
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a[i] / b[i];
    }
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator/(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(b != 0);
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a / b[i];
    }
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator/(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(b != 0);
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] / b;
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator%(const ac::array<T, N>& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ERRCHK(a.size() == b.size());
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] % b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator%(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a % b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator%(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = a[i] % b;
    return c;
}

template <typename T, typename U, size_t N>
bool __host__ __device__
operator==(const ac::array<T, N>& a, const ac::array<U, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    ERRCHK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

template <typename T, typename U, size_t N>
bool __host__ __device__
operator>=(const ac::array<T, N>& a, const ac::array<U, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    ERRCHK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] < b[i])
            return false;
    return true;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const ac::array<T, N>& a)
{
    static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
    ac::array<T, N> c;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = -a[i];
    return c;
}

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const ac::array<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.size(); ++i)
        os << obj[i] << (i + 1 < obj.size() ? ", " : "}");
    return os;
}

template <typename T, size_t N>
T __host__ __device__
prod(const ac::array<T, N> arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result = 1;
    for (size_t i = 0; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}
