#pragma once

#include "type_conversion.h"

#if defined(CUDA_ENABLED)
#include <cuda/std/array>
namespace ac {
template <typename T, size_t N> using base_array = cuda::std::array<T, N>;
}
#else
#include <array>
namespace ac {
template <typename T, size_t N> using base_array = std::array<T, N>;
}
#endif

// Mask __host__ __device__ directives if device runtime APIs are not available
#if !defined(DEVICE_ENABLED)
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
    // ac::array<int, 3> a{1,2,3}
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

    __host__ ac::array<T, N> reversed() const
    {
        ac::array<T, N> out{};
        for (size_t i{0}; i < N; ++i)
            out.resource[i] = resource[N - 1 - i];
        return out;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator+(const ac::array<T, N>& a,
                                                         const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator+(const T& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator+(const ac::array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator-(const ac::array<T, N>& a,
                                                         const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator-(const T& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator-(const ac::array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator*(const ac::array<T, N>& a,
                                                         const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator*(const T& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator*(const ac::array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator/(const ac::array<T, N>& a,
                                                         const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
        }
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator/(const T& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a / b[i];
        }
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator/(const ac::array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] / b;
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator%(const ac::array<T, N>& a,
                                                         const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator%(const T& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U>
    friend __host__ __device__ ac::array<T, N> operator%(const ac::array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U>
    friend __host__ __device__ bool operator==(const ac::array<T, N>& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < N; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    template <typename U>
    friend __host__ __device__ bool operator>=(const ac::array<T, N>& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < N; ++i)
            if (a[i] < b[i])
                return false;
        return true;
    }

    template <typename U>
    friend __host__ __device__ bool operator<=(const ac::array<T, N>& a, const ac::array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < N; ++i)
            if (a[i] > b[i])
                return false;
        return true;
    }

    friend __host__ __device__ ac::array<T, N> operator-(const ac::array<T, N>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        ac::array<T, N> c{};
        for (size_t i{0}; i < N; ++i)
            c[i] = -a[i];
        return c;
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const ac::array<T, N>& obj)
    {
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }
};

template <size_t N> using shape = ac::array<uint64_t, N>;
} // namespace ac

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

template <typename T, size_t N>
[[nodiscard]] __host__ __device__ T
prod(const ac::array<T, N>& arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result{1};
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

template <typename T, size_t N, typename U>
__host__ __device__ T
dot(const ac::array<T, N>& a, const ac::array<U, N>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    T res{0};
    for (size_t i{0}; i < N; ++i)
        res += a[i] * b[i];
    return res;
}

void test_array(void);
