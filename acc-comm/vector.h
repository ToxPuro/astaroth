#pragma once

#include <iostream>
#include <vector>

#include "errchk.h"

template <typename T, typename U>
std::vector<T>
operator+(const std::vector<T>& a, const std::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator+(const T& a, const std::vector<U>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(b.size());
    for (size_t i{0}; i < b.size(); ++i)
        c[i] = a + b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator+(const std::vector<T>& a, const U& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] + b;
    return c;
}

template <typename T, typename U>
std::vector<T>
operator-(const std::vector<T>& a, const std::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] - b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator-(const T& a, const std::vector<U>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(b.size());
    for (size_t i{0}; i < b.size(); ++i)
        c[i] = a - b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator-(const std::vector<T>& a, const U& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] - b;
    return c;
}

template <typename T, typename U>
std::vector<T>
operator*(const std::vector<T>& a, const std::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] * b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator*(const T& a, const std::vector<U>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(b.size());
    for (size_t i{0}; i < b.size(); ++i)
        c[i] = a * b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator*(const std::vector<T>& a, const U& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] * b;
    return c;
}

template <typename T, typename U>
std::vector<T>
operator/(const std::vector<T>& a, const std::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a[i] / b[i];
    }
    return c;
}

template <typename T, typename U>
std::vector<T>
operator/(const T& a, const std::vector<U>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    ERRCHK(b != 0);
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i) {
        ERRCHK(b[i] != 0);
        c[i] = a / b[i];
    }
    return c;
}

template <typename T, typename U>
std::vector<T>
operator/(const std::vector<T>& a, const U& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    ERRCHK(b != 0);
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] / b;
    return c;
}

template <typename T, typename U>
std::vector<T>
operator%(const std::vector<T>& a, const std::vector<U>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] % b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator%(const T& a, const std::vector<U>& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(b.size());
    for (size_t i{0}; i < b.size(); ++i)
        c[i] = a % b[i];
    return c;
}

template <typename T, typename U>
std::vector<T>
operator%(const std::vector<T>& a, const U& b)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] % b;
    return c;
}

// template <typename T, typename U>
// bool
// operator==(const std::vector<T>& a, const std::vector<U>& b)
// {
//     ERRCHK(a.size() == b.size());
//     static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
//     static_assert(std::is_same_v<T, U>,
//                   "Operator not enabled for parameters of different types. Perform an "
//                   "explicit cast such that both operands are of the same type");

//     for (size_t i{0}; i < a.size(); ++i)
//         if (a[i] != b[i])
//             return false;
//     return true;
// }

template <typename T, typename U>
bool
operator>=(const std::vector<T>& a, const std::vector<U>& b)
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

template <typename T, typename U>
bool
operator<=(const std::vector<T>& a, const std::vector<U>& b)
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

template <typename T>
std::vector<T>
operator-(const std::vector<T>& a)
{
    static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
    std::vector<T> c(a.size());
    for (size_t i{0}; i < a.size(); ++i)
        c[i] = -a[i];
    return c;
}

template <typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& obj)
{
    os << "{ ";
    for (const auto& elem : obj)
        os << elem << " ";
    os << "}";
    return os;
}

void test_vector(void);
