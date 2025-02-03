/**
 * A vector class with support for arithmetic operations and additional error checking.
 * Extends std::vector.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include "errchk.h"
#include "type_conversion.h"

namespace ac {

template <typename T> class vector {
  private:
    std::vector<T> m_resource;

  public:
    // Vector-like constructor
    // ac::vector<int> a(10, 1)
    explicit vector(const size_t count, const T& fill_value = 0)
        : m_resource(count, fill_value)
    {
    }

    // Construct from C-style count and pointer
    // ac::vector(count, ptr)
    explicit vector(const size_t count, const T* arr)
        : vector(count)
    {
        ERRCHK(count > 0);
        ERRCHK(arr);
        std::copy_n(arr, count, m_resource.begin());
    }

    // Construct from std::vector
    // ac::vector(std::vector{1,2,3})
    // ac::vector{{1,2,3}}
    explicit vector(const std::vector<T>& vec)
        : m_resource{vec}
    {
    }

    // Initializer list constructor
    // ac::vector<int> a{1,2,3}
    explicit vector(const std::initializer_list<T>& init_list)
        : m_resource{init_list}
    {
    }

    // Enable the subscript[] operator
    T& operator[](const size_t i)
    {
        ERRCHK(i < m_resource.size());
        return m_resource[i];
    }

    const T& operator[](const size_t i) const
    {
        ERRCHK(i < m_resource.size());
        return m_resource[i];
    }

    size_t size() const { return m_resource.size(); }

    auto data() const { return m_resource.data(); }
    auto data() { return m_resource.data(); }

    auto begin() const { return m_resource.begin(); }
    auto begin() { return m_resource.begin(); }

    auto end() const { return m_resource.end(); }
    auto end() { return m_resource.end(); }

    // auto get() { return m_resource; }
    // auto get() const { return m_resource; }

    template <typename U>
    friend ac::vector<T> operator+(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator+(const T& a, const ac::vector<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(b.size());
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator+(const ac::vector<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    friend ac::vector<T> operator-(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator-(const T& a, const ac::vector<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(b.size());
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator-(const ac::vector<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    friend ac::vector<T> operator*(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator*(const T& a, const ac::vector<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(b.size());
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator*(const ac::vector<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    friend ac::vector<T> operator/(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U> friend ac::vector<T> operator/(const T& a, const ac::vector<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U> friend ac::vector<T> operator/(const ac::vector<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        if constexpr (std::is_integral_v<U>)
            ERRCHK(b != 0);
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i) {
            c[i] = a[i] / b;
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(!std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U>
    friend ac::vector<T> operator%(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator%(const T& a, const ac::vector<U>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(b.size());
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U> friend ac::vector<T> operator%(const ac::vector<T>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U> friend bool operator==(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    template <typename U> friend bool operator>=(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < a.size(); ++i)
            if (a[i] < b[i])
                return false;
        return true;
    }

    template <typename U> friend bool operator<=(const ac::vector<T>& a, const ac::vector<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < a.size(); ++i)
            if (a[i] > b[i])
                return false;
        return true;
    }

    friend ac::vector<T> operator-(const ac::vector<T>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        ac::vector<T> c(a.size());
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = -a[i];
        return c;
    }

    friend std::ostream& operator<<(std::ostream& os, const ac::vector<T>& obj)
    {
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }
};

} // namespace ac

template <typename T>
[[nodiscard]] T
prod(const ac::vector<T>& arr)
{
    T result{1};
    for (size_t i{0}; i < arr.size(); ++i)
        result *= arr[i];
    return result;
}

/** Element-wise multiplication of vectors a and b */
template <typename T>
[[nodiscard]] ac::vector<T>
mul(const ac::vector<T>& a, const ac::vector<T>& b)
{
    ERRCHK(a.size() == b.size());

    ac::vector<T> c(a.size());

    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] * b[i];

    return c;
}

template <typename T, typename U>
[[nodiscard]] T
dot(const ac::vector<T>& a, const ac::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    T result{0};
    for (size_t i{0}; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

template <typename T, typename U>
[[nodiscard]] ac::vector<T>
concat(const ac::vector<T>& a, const ac::vector<U>& b)
{
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c{a};
    c.insert(c.end(), b.begin(), b.end());
    return c;
}

template <typename T, typename U>
[[nodiscard]] ac::vector<T>
concat(const ac::vector<T>& a, const U& b)
{
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    return concat(a, ac::vector<U>{b});
}
template <typename T, typename U>
[[nodiscard]] ac::vector<T>
concat(const T& a, const ac::vector<U>& b)
{
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    return concat(ac::vector<T>{a}, b);
}

template <typename T>
[[nodiscard]] ac::vector<T>
make_vector(const size_t count, const T* arr)
{
    return ac::vector<T>(count, arr);
}

void test_vector(void);
