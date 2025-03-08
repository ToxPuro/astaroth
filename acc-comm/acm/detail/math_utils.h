#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

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
