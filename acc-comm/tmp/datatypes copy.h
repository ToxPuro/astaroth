#pragma once
#include <iostream>

#include "type_conversion.h"

#include "buffer.h"

#if defined(CUDA_ENABLED)
#include <cuda/std/array>
namespace ac {
template <typename T, size_t N> using base_array = cuda::std::array<T, N>;
}
#else
namespace ac {
template <typename T, size_t N> using base_array = std::array<T, N>;
}
#endif

#if defined(DEVICE_ENABLED)
#else
#define __host__
#define __device__
#endif

// Disable errchecks in device code (not supported as of 2024-11-11)
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1)
#undef ERRCHK
#define ERRCHK(expr)
#undef ERRCHK_EXPR_DESC
#define ERRCHK_EXPR_DESC(expr, ...)
#endif

namespace ac {
template <typename T, size_t N> class array {
  private:
    ac::base_array<T, N> resource{};

  public:
    // Default constructor
    __host__ __device__ array() = default;

    // Initializer list constructor
    // StaticArray<int, 3> a{1,2,3}
    __host__ __device__ array(const std::initializer_list<T>& init_list)
    {
        ERRCHK(init_list.size() == N);
        std::copy(init_list.begin(), init_list.end(), resource.begin());
    }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](const size_t i)
    {
        ERRCHK(i < N);
        return resource[i];
    }
    __host__ __device__ const T& operator[](const size_t i) const
    {
        ERRCHK(i < N);
        return resource[i];
    }
    __host__ __device__ auto size() const { return resource.size(); }
    auto begin() { return resource.begin(); }
    auto begin() const { return resource.begin(); }
    auto end() { return resource.end(); }
    auto end() const { return resource.end(); }
    auto data() { return resource.data(); }
    auto data() const { return resource.data(); }
};

template <size_t N> using shape = ac::array<uint64_t, N>;
} // namespace ac

template <size_t N> using Index     = ac::array<uint64_t, N>;
template <size_t N> using Shape     = ac::array<uint64_t, N>;
template <size_t N> using Direction = ac::array<int64_t, N>;
template <size_t N> using MPIIndex  = ac::array<int, N>;
template <size_t N> using MPIShape  = ac::array<int, N>;
using UserType                      = double;

template <typename T, size_t N>
[[nodiscard]] auto
zeros()
{
    ac::array<T, N> arr{};
    std::fill_n(arr.begin(), N, as<T>(0));
    return arr;
};

template <typename T, size_t N>
[[nodiscard]] auto
ones()
{
    ac::array<T, N> arr{};
    std::fill_n(arr.begin(), N, as<T>(1));
    return arr;
};

template <typename T, size_t N>
[[nodiscard]] auto
fill(const T& fill_value)
{
    ac::array<T, N> arr{};
    std::fill_n(arr.begin(), N, fill_value);
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
//     for (size_t i{0}; i < count; ++i)
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
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator+(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator+(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
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
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator-(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
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
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator*(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator*(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
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
    for (size_t i{0}; i < c.size(); ++i) {
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
    for (size_t i{0}; i < c.size(); ++i) {
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
    for (size_t i{0}; i < c.size(); ++i)
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
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a[i] % b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator%(const T& a, const ac::array<T, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = a % b[i];
    return c;
}

template <typename T, size_t N>
ac::array<T, N> __host__ __device__
operator%(const ac::array<T, N>& a, const T& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");

    ac::array<T, N> c;
    for (size_t i{0}; i < c.size(); ++i)
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
    for (size_t i{0}; i < a.size(); ++i)
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
    for (size_t i{0}; i < a.size(); ++i)
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
    for (size_t i{0}; i < c.size(); ++i)
        c[i] = -a[i];
    return c;
}

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const ac::array<T, N>& obj)
{
    os << "{";
    for (size_t i{0}; i < obj.size(); ++i)
        os << obj[i] << (i + 1 < obj.size() ? ", " : "}");
    return os;
}

template <typename T, size_t N>
T __host__ __device__
prod(const ac::array<T, N>& arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result{1};
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

void test_datatypes(void);
