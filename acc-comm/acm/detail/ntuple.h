#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>
#include <vector>

#include "cuda_utils.h"
#include "errchk.h"

namespace ac {

template <typename T, size_t N> class static_ntuple;

template <typename T> class ntuple {
  private:
    std::vector<T> m_resource;

  public:
    explicit ntuple(const std::initializer_list<T>& init_list)
        : m_resource{init_list}
    {
    }

    explicit ntuple(const std::vector<T>& vec)
        : m_resource{vec}
    {
    }

    template <size_t N>
    explicit ntuple(const ac::static_ntuple<T, N>& in)
        : m_resource(in.begin(), in.end())
    {
    }

    auto size() const { return m_resource.size(); }

    auto data() const { return m_resource.data(); }
    auto data() { return m_resource.data(); }

    auto begin() const { return m_resource.begin(); }
    auto begin() { return m_resource.begin(); }

    auto end() const { return m_resource.end(); }
    auto end() { return m_resource.end(); }

    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_resource[i];
    }

    const T& operator[](const size_t i) const
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

    T back() const { return m_resource[size() - 1]; }
};

template <typename T>
[[nodiscard]] ac::ntuple<T>
make_ntuple(const size_t count, const T& fill_value)
{
    return ac::ntuple<T>{std::vector<T>(count, fill_value)};
}

template <typename T>
[[nodiscard]] ac::ntuple<T>
make_ntuple_from_ptr(const size_t count, const T* data)
{
    ERRCHK(count > 0);
    ERRCHK(data);
    ac::ntuple<T> retval{make_ntuple<T>(count, 0)};
    std::copy_n(data, count, retval.begin());
    return retval;
}

template <typename T>
[[nodiscard]] ac::ntuple<T>
slice(const ac::ntuple<T>& ntuple, const size_t lb, const size_t ub)
{
    ERRCHK(lb < ub);
    ac::ntuple<T> out{ac::make_ntuple<T>(ub - lb, 0)};
    for (size_t i{lb}; i < ub; ++i)
        out[i - lb] = ntuple[i];
    return out;
}

template <typename T>
[[nodiscard]] T
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
[[nodiscard]] T
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
[[nodiscard]] ac::ntuple<T>
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

template <typename T>
[[nodiscard]] bool
within_box(const ac::ntuple<T>& coords, const ac::ntuple<T>& box_dims,
           const ac::ntuple<T>& box_offset)
{
    ERRCHK(coords.size() == box_dims.size());
    ERRCHK(coords.size() == box_offset.size());
    for (size_t i{0}; i < coords.size(); ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

} // namespace ac

namespace ac {

template <typename T, size_t N> class static_ntuple {
  private:
    size_t m_count;
    T      m_resource[N];

  public:
    __host__ __device__ explicit static_ntuple(const std::initializer_list<T>& init_list)
        : m_count{init_list.size()}
    {
        ERRCHK(init_list.size() <= N);
        std::copy(init_list.begin(), init_list.end(), m_resource);
    }

    __host__ __device__ explicit static_ntuple(const std::vector<T>& vec)
        : m_count{vec.size()}
    {
        ERRCHK(vec.size() <= N);
        std::copy(vec.begin(), vec.end(), m_resource);
    }

    __host__ __device__ explicit static_ntuple(const ac::ntuple<T>& in)
        : m_count{in.size()}
    {
        ERRCHK(in.size() <= N);
        std::copy(in.begin(), in.end(), m_resource);
    }

    __host__ __device__ auto size() const { return m_count; }

    __host__ __device__ auto data() const { return m_resource; }
    __host__ __device__ auto data() { return m_resource; }

    __host__ __device__ auto begin() const { return data(); }
    __host__ __device__ auto begin() { return data(); }

    __host__ __device__ auto end() const { return data() + size(); }
    __host__ __device__ auto end() { return data() + size(); }

    __host__ __device__ T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_resource[i];
    }

    __host__ __device__ const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_resource[i];
    }

    __host__ friend std::ostream& operator<<(std::ostream& os, const static_ntuple<T, N>& obj)
    {
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator+(const ac::static_ntuple<T, N>& a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator+(const T&                       a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a + b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator+(const ac::static_ntuple<T, N>& a,
                                                                 const U&                       b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] + b;
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator-(const ac::static_ntuple<T, N>& a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator-(const T&                       a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a - b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator-(const ac::static_ntuple<T, N>& a,
                                                                 const U&                       b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] - b;
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator*(const ac::static_ntuple<T, N>& a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator*(const T&                       a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a * b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator*(const ac::static_ntuple<T, N>& a,
                                                                 const U&                       b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] * b;
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator/(const ac::static_ntuple<T, N>& a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a[i] / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator/(const T&                       a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            if constexpr (std::is_integral_v<U>)
                ERRCHK(b[i] != 0);
            c[i] = a / b[i];
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator/(const ac::static_ntuple<T, N>& a,
                                                                 const U&                       b)
    {
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        if constexpr (std::is_integral_v<U>)
            ERRCHK(b != 0);
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i) {
            c[i] = a[i] / b;
            if constexpr (std::is_floating_point_v<T>)
                ERRCHK(!std::isnormal(c[i]));
        }
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator%(const ac::static_ntuple<T, N>& a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        ERRCHK(a.size() == b.size());
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");

        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator%(const T&                       a,
                                                                 const ac::static_ntuple<U, N>& b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{b};
        for (size_t i{0}; i < b.size(); ++i)
            c[i] = a % b[i];
        return c;
    }

    template <typename U>
    __host__ __device__ friend ac::static_ntuple<T, N> operator%(const ac::static_ntuple<T, N>& a,
                                                                 const U&                       b)
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = a[i] % b;
        return c;
    }

    template <typename U>
    __host__ __device__ friend bool operator==(const ac::static_ntuple<T, N>& a,
                                               const ac::static_ntuple<U, N>& b)
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

    template <typename U>
    __host__ __device__ friend bool operator>=(const ac::static_ntuple<T, N>& a,
                                               const ac::static_ntuple<U, N>& b)
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

    template <typename U>
    __host__ __device__ friend bool operator<=(const ac::static_ntuple<T, N>& a,
                                               const ac::static_ntuple<U, N>& b)
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

    template <typename U>
    __host__ __device__ friend bool operator<(const ac::static_ntuple<T, N>& a,
                                              const ac::static_ntuple<U, N>& b)
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

    template <typename U>
    __host__ __device__ friend bool operator>(const ac::static_ntuple<T, N>& a,
                                              const ac::static_ntuple<U, N>& b)
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

    friend ac::static_ntuple<T, N> operator-(const ac::static_ntuple<T, N>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        ac::static_ntuple<T, N> c{a};
        for (size_t i{0}; i < a.size(); ++i)
            c[i] = -a[i];
        return c;
    }
};

template <typename T, size_t N>
[[nodiscard]] ac::static_ntuple<T, N>
make_static_ntuple(const size_t count, const T& fill_value)
{
    return ac::static_ntuple<T, N>{ac::make_ntuple(count, fill_value)};
}

template <typename T, size_t N>
[[nodiscard]] ac::static_ntuple<T, N>
make_static_ntuple_from_ptr(const size_t count, const T* data)
{
    return ac::static_ntuple<T, N>{ac::make_ntuple_from_ptr(count, data)};
}

template <typename T, size_t N>
[[nodiscard]] ac::static_ntuple<T, N>
slice(const ac::static_ntuple<T, N>& static_ntuple, const size_t lb, const size_t ub)
{
    return ac::static_ntuple<T, N>{ac::slice(ac::ntuple{static_ntuple}, lb, ub)};
}

template <typename T, size_t N>
[[nodiscard]] __host__ __device__ T
prod(const ac::static_ntuple<T, N>& in)
{
    T out{1};
    for (size_t i{0}; i < in.size(); ++i)
        out *= in[i];
    return out;
}

/** Element-wise multiplication of static_ntuples a and b */
template <typename T, typename U, size_t N>
[[nodiscard]] __host__ __device__ ac::static_ntuple<T, N>
mul(const ac::static_ntuple<T, N>& a, const ac::static_ntuple<U, N>& b)
{
    ERRCHK(a.size() == b.size());
    static_assert(std::is_same_v<T, U>,
                  "Operator not enabled for parameters of different types. Perform an "
                  "explicit cast such that both operands are of the same type");

    ac::static_ntuple<T, N> c{a};

    for (size_t i{0}; i < a.size(); ++i)
        c[i] = a[i] * b[i];

    return c;
}

template <typename T, typename U, size_t N>
[[nodiscard]] __host__ __device__ T
dot(const ac::static_ntuple<T, N>& a, const ac::static_ntuple<U, N>& b)
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

template <typename T, size_t N>
[[nodiscard]] __host__ __device__ T
to_linear(const ac::static_ntuple<T, N>& coords, const ac::static_ntuple<T, N>& shape)
{
    ERRCHK(coords.size() == shape.size());
    T result{0};
    for (size_t j{0}; j < shape.size(); ++j) {
        T factor{1};
        for (size_t i{0}; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

template <typename T, size_t N>
[[nodiscard]] __host__ __device__ ac::static_ntuple<T, N>
                                  to_spatial(const T index, const ac::static_ntuple<T, N>& shape)
{
    ac::static_ntuple<T, N> coords{shape};
    for (size_t j{0}; j < shape.size(); ++j) {
        T divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T, size_t N>
[[nodiscard]] bool
within_box(const ac::static_ntuple<T, N>& coords, const ac::static_ntuple<T, N>& box_dims,
           const ac::static_ntuple<T, N>& box_offset)
{
    ERRCHK(coords.size() == box_dims.size());
    ERRCHK(coords.size() == box_offset.size());
    for (size_t i{0}; i < coords.size(); ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

} // namespace ac

// Type aliases
namespace ac {
using index     = ac::ntuple<uint64_t>;
using shape     = ac::ntuple<uint64_t>;
using direction = ac::ntuple<int64_t>;

index     make_index(const size_t count, const uint64_t& fill_value);
shape     make_shape(const size_t count, const uint64_t& fill_value);
direction make_direction(const size_t count, const int64_t& fill_value);
} // namespace ac

// Testing functions
void test_ntuple();
