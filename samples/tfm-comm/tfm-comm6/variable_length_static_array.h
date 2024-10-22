#pragma once
#include <cstddef>

#include <iostream>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
#define __host__
#define __device__
#endif

#include "errchk.h"
#include "type_conversion.h"

template <typename T, size_t N> struct VariableLengthStaticArray {
    size_t count;
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    // Default constructor (disabled)
    // __host__ __device__ VariableLengthStaticArray() : count(0), data{} {}

    // Vector-like constructor
    // VariableLengthStaticArray<int, N> a(10, 1)
    __host__ __device__ VariableLengthStaticArray(const size_t count, const T& fill_value = 0)
        : count(count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        for (size_t i = 0; i < count; ++i)
            data[i] = fill_value;
    }

    // Initializer list constructor
    // VariableLengthStaticArray<int, 3> a = {1,2,3}
    __host__ __device__ VariableLengthStaticArray(const std::initializer_list<T>& init_list)
        : count(init_list.size())
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        std::copy(init_list.begin(), init_list.begin() + count, data);
    }

    // Copy constructor with proper casting
    // VariableLengthStaticArray<T, N> a(VariableLengthStaticArray<U, N> b)
    template <typename U>
    __host__ __device__ VariableLengthStaticArray(const VariableLengthStaticArray<U, N>& other)
        : count(other.count)
    {
        for (size_t i = 0; i < count; ++i)
            data[i] = as<T>(other.data[i]);
    }

    // Construct from a pointer
    __host__ __device__ VariableLengthStaticArray(const size_t count, const T* arr) : count(count)
    {
        ERRCHK(count > 0);
        ERRCHK(count <= N);
        ERRCHK(arr);
        for (size_t i = 0; i < count; ++i)
            data[i] = arr[i];
    }

    // Record the number of elements
    __host__ __device__ constexpr size_t capacity(void) const { return N; }

    // Enable the array[] operator
    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }
};

template <typename T, size_t N>
__host__ std::ostream&
operator<<(std::ostream& os, const VariableLengthStaticArray<T, N>& obj)
{
    os << "{";
    for (size_t i = 0; i < obj.count; ++i)
        os << obj[i] << (i + 1 < obj.count ? ", " : "}");
    return os;
}
