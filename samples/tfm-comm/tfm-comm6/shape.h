#pragma once

#include "static_array.h"

constexpr size_t MAX_NDIMS = 3;
typedef StaticArray<uint64_t, MAX_NDIMS> Index;
typedef StaticArray<uint64_t, MAX_NDIMS> Shape;
typedef StaticArray<int, MAX_NDIMS> MPIIndex;
typedef StaticArray<int, MAX_NDIMS> MPIShape;

// struct Shape : public StaticArray<uint64_t, MAX_NDIMS> {

//     // Initialize Shape to 1 by default
//     __host__ __device__ Shape() : StaticArray<uint64_t, MAX_NDIMS>(1) {}

//     // Inherit the rest of the functions from StaticArray
//     using StaticArray<uint64_t, MAX_NDIMS>::StaticArray;
// };
// typedef StaticArray<uint64_t, MAX_NDIMS> Index;
