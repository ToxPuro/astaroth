#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

uint64_t prod(const size_t count, const uint64_t* arr);

template <typename T>
bool
within_machine_epsilon(const T& a, const T& b)
{
    const auto epsilon{std::numeric_limits<T>::epsilon()};
    return (a >= b - epsilon) && (a <= b + epsilon);
}

void test_math_utils(void);
