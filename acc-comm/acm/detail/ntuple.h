#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "errchk.h"

namespace ac {

template <typename T> class ntuple {
  private:
    std::vector<T> m_resource;

  public:
    ntuple(const std::initializer_list<T>& init_list)
        : m_resource{init_list}
    {
    }

    ntuple(const std::vector<T>& vec)
        : m_resource{vec}
    {
    }

    auto size() const { return m_resource.size(); }

    auto data() const { return m_resource.data(); }
    auto data() { return m_resource.data(); }

    auto begin() const { return m_resource.begin(); }
    auto begin() { return m_resource.begin(); }

    auto end() const { return m_resource.end(); }
    auto end() { return m_resource.end(); }

    auto& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_resource[i];
    }

    const auto& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_resource[i];
    }

    friend std::ostream& operator<<(std::ostream& os, const ntuple& obj)
    {
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }

    template <typename U>
    friend ac::ntuple<T> operator+(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator+(const T& a, const ac::ntuple<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator+(const ac::ntuple<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    friend ac::ntuple<T> operator-(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator-(const T& a, const ac::ntuple<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator-(const ac::ntuple<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    friend ac::ntuple<T> operator*(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator*(const T& a, const ac::ntuple<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator*(const ac::ntuple<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    friend ac::ntuple<T> operator/(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator/(const T& a, const ac::ntuple<U>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator/(const ac::ntuple<T>& a, const U& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        if constexpr (std::is_integral_v<U>)
            ERRCHK(b != 0);
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            c[i] = a[i] / b;
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(!std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U>
    friend ac::ntuple<T> operator%(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator%(const T& a, const ac::ntuple<U>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U> friend ac::ntuple<T> operator%(const ac::ntuple<T>& a, const U& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U> friend bool operator==(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
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

    template <typename U> friend bool operator>=(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
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

    template <typename U> friend bool operator<=(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
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

    template <typename U> friend bool operator<(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < a.size(); ++i)
            if (a[i] >= b[i])
                return false;
        return true;
    }

    template <typename U> friend bool operator>(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        for (size_t i{0}; i < a.size(); ++i)
            if (a[i] <= b[i])
                return false;
        return true;
    }

    friend ac::ntuple<T> operator-(const ac::ntuple<T>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        ac::ntuple<T> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = -a[i];
        return c;
    }

    auto back() const { return m_resource[size() - 1]; }
};

template <typename T>
[[nodiscard]] auto
make_ntuple(const size_t count, const T& fill_value = 0)
{
    return ac::ntuple<T>{std::vector<T>(count, fill_value)};
}

template <typename T>
[[nodiscard]] auto
make_ntuple_from_ptr(const size_t count, const T* data)
{
    ERRCHK(count > 0);
    ERRCHK(data);
    ac::ntuple<T> retval{make_ntuple<T>(count)};
    std::copy_n(data, count, retval.begin());
    return retval;
}

template <typename T>
[[nodiscard]] auto
slice(const ac::ntuple<T>& ntuple, const size_t lb, const size_t ub)
{
    ERRCHK(lb < ub);
    ac::ntuple<T> out{ac::make_ntuple<T>(ub - lb)};
    for (size_t i{lb}; i < ub; ++i)
        out[i - lb] = ntuple[i];
    return out;
}

template <typename T>
[[nodiscard]] auto
prod(const ac::ntuple<T>& in)
{
    T out{1};
    for (size_t i{0}; i < in.size(); ++i)
        out *= in[i];
    return out;
}

/** Element-wise multiplication of ntuples a and b */
template <typename T, typename U>
[[nodiscard]] ac::ntuple<T>
mul(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    ac::ntuple<T> c{a};

    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] * b[i];

    return c;
}

template <typename T, typename U>
[[nodiscard]] T
dot(const ac::ntuple<T>& a, const ac::ntuple<U>& b)
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

template <typename T>
[[nodiscard]] auto
to_linear(const ac::ntuple<T>& coords, const ac::ntuple<T>& shape)
{
    T result{0};
    for (size_t j{0}; j < shape.size(); ++j) {
        T factor{1};
        for (size_t i{0}; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <typename T>
[[nodiscard]] auto
to_spatial(const T index, const ac::ntuple<T>& shape)
{
    ac::ntuple<T> coords{shape};
    for (size_t j{0}; j < shape.size(); ++j) {
        T divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

} // namespace ac
