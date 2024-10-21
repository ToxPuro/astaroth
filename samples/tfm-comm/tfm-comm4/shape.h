#pragma once
#include "static_array.h"

constexpr size_t MAX_NDIMS = 4;
typedef StaticArray<uint64_t, MAX_NDIMS> Shape;
typedef StaticArray<uint64_t, MAX_NDIMS> Index;

typedef StaticArray<int, MAX_NDIMS> MPIIndex;
typedef StaticArray<int, MAX_NDIMS> MPIShape;
