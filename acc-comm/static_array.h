#pragma once
#include <cstddef>

#include <iostream>
#include <type_traits>

#include "cuda_utils.h"
#include "errchk.h"
#include "type_conversion.h"

#include "vector.h"

namespace ac {

template <typename T, size_t N> class static_array {
  private:
    size_t count;
    T data[N]{};

  public:
    // Record the number of elements
    __host__ __device__ constexpr size_t capacity(void) const { return N; }

    // Enable the subscript[] operator
    __host__ __device__ T& operator[](const size_t i)
    {
        ERRCHK(i < count);
        return data[i];
    }
    __host__ __device__ const T& operator[](const size_t i) const
    {
        ERRCHK(i < count);
        return data[i];
    }

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor (disabled)
    // __host__ __device__ static_array() : count(0), data{} {}

    // Vector-like constructor
    // static_array<int, N> a(10, 1)
    __host__ __device__ static_array(const size_t in_count, const T& fill_value = 0)
        : count(in_count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        for (size_t i{0}; i < count; ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // static_array<int, 3> a{1,2,3}
    __host__ __device__ static_array(const std::initializer_list<T>& init_list)
        : count(init_list.size())
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        std::copy(init_list.begin(), init_list.begin() + count, data);
    }

    // Copy constructor with proper casting
    // static_array<T, N> a(static_array<U, N> b)
    template <typename U>
    __host__ __device__ explicit static_array(const static_array<U, N>& other)
        : count(other.count)
    {
        for (size_t i{0}; i < count; ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ explicit static_array(const size_t in_count, const T* arr)
        : count(in_count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        ERRCHK(arr);
        for (size_t i{0}; i < count; ++i)
            data[i] = arr[i];
    }

    // Construct from a vector
    // template<typename VectorType>
    __host__ __device__ static_array(const ac::vector<T>& vec)
        : static_array(vec.size())
    {
        for (size_t i{0}; i < vec.size(); ++i)
            data[i] = vec[i];
    }

    __host__ __device__ static_array(const std::vector<T>& vec)
        : static_array(vec.size())
    {
        for (size_t i{0}; i < vec.size(); ++i)
            data[i] = vec[i];
    }

    __host__ __device__ size_t size() const { return count; }

    // Common operations
    template <typename U> T __host__ __device__ dot(const static_array<U, N> other) const
    {
        static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
        static_assert(std::is_same_v<T, U>,
                      "Operator not enabled for parameters of different types. Perform an "
                      "explicit cast such that both operands are of the same type");
        T res = 0;
        for (size_t i{0}; i < count; ++i)
            res += data[i] * other[i];
        return res;
    }

    __host__ static_array<T, N> reversed() const
    {
        static_array<T, N> out(count);
        for (size_t i{0}; i < count; ++i)
            out.data[i] = data[count - 1 - i];
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
        ERRCHK(a.count == b.count);
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(b.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        ERRCHK(a.count == b.count);
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(b.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        ERRCHK(a.count == b.count);
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(b.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        ERRCHK(a.count == b.count);
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i) {
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
        static_array<T, N> c(b.count);
        for (size_t i{0}; i < c.count; ++i) {
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
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        ERRCHK(a.count == b.count);
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(b.count);
        for (size_t i{0}; i < c.count; ++i)
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
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
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
        ERRCHK(a.count == b.count);
        for (size_t i{0}; i < a.count; ++i)
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
        ERRCHK(a.count == b.count);
        for (size_t i{0}; i < a.count; ++i)
            if (a[i] < b[i])
                return false;
        return true;
    }

    friend static_array<T, N> __host__ __device__ operator-(const static_array<T, N>& a)
    {
        static_assert(std::is_signed_v<T>, "Operator enabled only for signed types");
        static_array<T, N> c(a.count);
        for (size_t i{0}; i < c.count; ++i)
            c[i] = -a[i];
        return c;
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const static_array<T, N>& obj)
    {
        os << "{";
        for (size_t i{0}; i < obj.count; ++i)
            os << obj[i] << (i + 1 < obj.count ? ", " : "}");
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
    for (size_t i{0}; i < arr.count; ++i)
        result *= arr[i];
    return result;
}

// template <typename T, size_t N>
// typename std::enable_if<std::is_arithmetic_v<T>, static_array<T, N>>::type __host__
// __device__ operator+(const static_array<T, N>& a, const static_array<T, N>& b)
// {
//     ERRCHK(a.count == b.count);
//     static_array<T, N> c(a.count);
//     for (size_t i{0}; i < c.count; ++i)
//         c[i] = a[i] + b[i];
//     return c;
// }

void test_static_array(void);
