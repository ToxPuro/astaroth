#pragma once

/*
USING:
    1. Use TG_DTYPE as the generic type in the prototype file
    2. Use TG_TYPE_GENERIC_FUNCTION(name) to declare generic functions
    3. Modify the MAPS function to choose which types should be generalized and to control the type
    of the decision variable X (type, pointer, or const pointer)
*/

// Type-specific functions
#define TG_CAT_(x, y) x##_##y
#define TG_CAT(x, y) TG_CAT_(x, y)
#define TG_TYPE_SPECIFIC_FUNCTION(name, type) TG_CAT(name, type)
#define TG_TYPE_GENERIC_FUNCTION(name) TG_CAT(name, TG_DTYPE)

// Type-specific declarations
#define TG_OUTPUT_HEADER
#define TG_DTYPE int
#include "tgarray_prototype.h"
#undef TG_DTYPE

#define TG_DTYPE double
#include "tgarray_prototype.h"
#undef TG_DTYPE
#undef TG_OUTPUT_HEADER

// Generic function mapping
#define MAP(name, type)                                                                            \
    type:                                                                                          \
    TG_TYPE_SPECIFIC_FUNCTION(name, type)

#define MAP_PTR(name, type) type* : TG_TYPE_SPECIFIC_FUNCTION(name, type)
#define MAP_CONST_PTR(name, type) MAP_PTR(name, type), const MAP_PTR(name, type)
#define MAPS(name, map) map(name, int), map(name, double)

// Declare generics
#define print(X) _Generic((X), MAPS(print, MAP))(X)
