#pragma once

#include "static_array.h"

constexpr size_t MAX_NDIMS = 4;

using Index     = StaticArray<uint64_t, MAX_NDIMS>;
using Shape     = StaticArray<uint64_t, MAX_NDIMS>;
using Direction = StaticArray<int64_t, MAX_NDIMS>;
using MPIIndex  = StaticArray<int, MAX_NDIMS>;
using MPIShape  = StaticArray<int, MAX_NDIMS>;
