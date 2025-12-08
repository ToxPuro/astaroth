#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <tuple>

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T>
T
vecprod(const std::vector<T>& vec)
{
    return std::reduce(vec.begin(), vec.end(), static_cast<T>(1), std::multiplies<T>());
}

template <typename T>
bool
within_machine_epsilon(const T& a, const T& b)
{
    const auto epsilon{std::numeric_limits<T>::epsilon()};
    return (a >= b - epsilon) && (a <= b + epsilon);
}

template <typename... Inputs>
bool
same_size(const Inputs&... inputs)
{
    const size_t count{std::get<0>(std::tuple(inputs...)).size()};
    return ((inputs.size() == count) && ...);
}

/** Returns true if the lines on intervals [a1, a2) and [b1, b2) intersect */
template <typename T>
bool
intersect_lines(const T a1, const T a2, const T b1, const T b2)
{
    static_assert(std::is_integral_v<T>, "Operator enabled only for integral types");
    return (a1 >= b1 && a1 < b2) || (b1 >= a1 && b1 < a2);
}
