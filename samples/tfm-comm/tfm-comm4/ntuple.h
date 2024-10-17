#pragma once
#include <cstddef>

template <typename T, size_t N> struct StaticArray {
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");
    constexpr size_t len(void) const { return N; }

    StaticArray(const StaticArray<T, N>& input) {}
};
