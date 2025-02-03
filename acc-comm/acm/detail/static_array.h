#pragma once
#include <cstddef>

#include <iostream>
#include <type_traits>

#include "cuda_utils.h"
#include "errchk.h"
#include "type_conversion.h"

namespace ac {

template <typename T, size_t N> class static_array {
  private:
    size_t m_count;
    T      m_data[N]{};

  public:
    // Record the number of elements
    __host__ __device__ constexpr size_t capacity(void) const { return N; }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](const size_t i)
    {
        ERRCHK(i < m_count);
        return m_data[i];
    }
    __host__ __device__ const T& operator[](const size_t i) const
    {
        ERRCHK(i < m_count);
        return m_data[i];
    }

    static_assert(sizeof(T) * N <= 2048,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor (disabled)
    // __host__ __device__ static_array() : m_count(0), m_data{} {}

    // Vector-like constructor
    // static_array<int, N> a(10, 1)
    __host__ __device__ static_array(const size_t count, const T& fill_value = 0)
        : m_count(count)
    {
        ERRCHK(m_count > 0);
        ERRCHK(m_count <= N);
        for (size_t i{0}; i < m_count; ++i)
            m_data[i] = fill_value;
    }

    // Initializer list constructor
    // static_array<int, 3> a{1,2,3}
    __host__ __device__ static_array(const std::initializer_list<T>& init_list)
        : m_count(init_list.size())
    {
        ERRCHK(m_count > 0);
        ERRCHK(m_count <= N);
        std::copy(init_list.begin(), init_list.begin() + m_count, m_data);
    }

    // Copy constructor with proper casting
    // static_array<T, N> a(static_array<U, N> b)
    // template <typename U>
    // __host__ __device__ explicit static_array(const static_array<U, N>& other)
    //     : m_count(other.m_count)
    // {
    //     for (size_t i{0}; i < m_count; ++i)
    //         m_data[i] = as<T>(other.m_data[i]);
    // }

    // Construct from a pointer
    __host__ __device__ explicit static_array(const size_t count, const T* arr)
        : m_count(count)
    {
        ERRCHK(m_count > 0);
        ERRCHK(m_count <= N);
        ERRCHK(arr);
        for (size_t i{0}; i < m_count; ++i)
            m_data[i] = arr[i];
    }

    // Construct from a vector
    // template<typename VectorType>
    // __host__ __device__ static_array(const ac::ntuple<T>& vec)
    //     : static_array(vec.size())
    // {
    //     for (size_t i{0}; i < vec.size(); ++i)
    //         m_data[i] = vec[i];
    // }

    // __host__ __device__ static_array(const std::vector<T>& vec)
    //     : static_array(vec.size())
    // {
    //     for (size_t i{0}; i < vec.size(); ++i)
    //         m_data[i] = vec[i];
    // }

    __host__ __device__ auto size() const { return m_count; }

    __host__ __device__ auto data() const { return m_data; }
    __host__ __device__ auto data() { return m_data; }

    __host__ __device__ auto begin() const { return data(); }
    __host__ __device__ auto begin() { return data(); }

    __host__ __device__ auto end() const { data() + size(); }
    __host__ __device__ auto end() { data() + size(); }

    // Common operations
    template <typename U> T __host__ __device__ dot(const static_array<U, N> other) const
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        T res = 0;
        for (size_t i{0}; i < m_count; ++i)
            res += m_data[i] * other[i];
        return res;
    }

    __host__ static_array<T, N> reversed() const
    {
        static_array<T, N> out(m_count);
        for (size_t i{0}; i < m_count; ++i)
            out.m_data[i] = m_data[m_count - 1 - i];
        return out;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator+(const static_array<T, N>& a,
                                                            const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator+(const T& a, const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(b.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator+(const static_array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator-(const static_array<T, N>& a,
                                                            const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator-(const T& a, const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(b.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator-(const static_array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator*(const static_array<T, N>& a,
                                                            const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator*(const T& a, const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(b.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator*(const static_array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator/(const static_array<T, N>& a,
                                                            const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
        }
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator/(const T& a, const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        static_array<T, N> c(b.m_count);
        for (size_t i{0}; i < c.m_count; ++i) {
            ERRCHK(b[i] != 0);
            c[i] = a / b[i];
        }
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator/(const static_array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(b != 0);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] / b;
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator%(const static_array<T, N>& a,
                                                            const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator%(const T& a, const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(b.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U>
    friend static_array<T, N> __host__ __device__ operator%(const static_array<T, N>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U>
    friend bool __host__ __device__ operator==(const static_array<T, N>& a,
                                               const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        for (size_t i{0}; i < a.m_count; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    template <typename U>
    friend bool __host__ __device__ operator>=(const static_array<T, N>& a,
                                               const static_array<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ERRCHK(a.m_count == b.m_count);
        for (size_t i{0}; i < a.m_count; ++i)
            if (a[i] < b[i])
                return false;
        return true;
    }

    friend static_array<T, N> __host__ __device__ operator-(const static_array<T, N>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        static_array<T, N> c(a.m_count);
        for (size_t i{0}; i < c.m_count; ++i)
            c[i] = -a[i];
        return c;
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const static_array<T, N>& obj)
    {
        os << "{";
        for (size_t i{0}; i < obj.m_count; ++i)
            os << obj[i] << (i + 1 < obj.m_count ? ", " : "}");
        return os;
    }
};

} // namespace ac

template <typename T, size_t N>
T __host__ __device__
prod(const ac::static_array<T, N> arr)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    T result = 1;
    for (size_t i{0}; i < arr.m_count; ++i)
        result *= arr[i];
    return result;
}

// template <typename T, size_t N>
// typename std::enable_if<std::is_arithmetic_v<T>, static_array<T, N>>::type __host__
// __device__ operator+(const static_array<T, N>& a, const static_array<T, N>& b)
// {
//     ERRCHK(a.m_count == b.m_count);
//     static_array<T, N> c(a.m_count);
//     for (size_t i{0}; i < c.m_count; ++i)
//         c[i] = a[i] + b[i];
//     return c;
// }

void test_static_array(void);
