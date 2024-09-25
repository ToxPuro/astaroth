#pragma once

/*
USING:
    1. Use TG_DTYPE as the generic type in the prototype file
    2. Use TG_TYPE_GENERIC_FUNCTION(name) to declare generic functions
    3. Modify the TG_MAPS function to choose which types should be generalized and to control the
type of the decision variable X (type, pointer, or const pointer)

    - Short: define TG_DTYPE0...7 depending on what is needed
*/

// Function name concatenation
#define TG_CAT_(x, y) x##_##y
#define TG_CAT(x, y) TG_CAT_(x, y)
#define TG_TYPE_SPECIFIC_FUNCTION(name, type) TG_CAT(name, type)
#define TG_TYPE_GENERIC_FUNCTION(name) TG_CAT(name, TG_DTYPE)

// Generic function mapping
#define TG_MAP(name, type)                                                                         \
    type:                                                                                          \
    TG_TYPE_SPECIFIC_FUNCTION(name, type)

#define TG_MAP_PTR(name, type) type* : TG_TYPE_SPECIFIC_FUNCTION(name, type)
#define TG_MAP_CONST_PTR(name, type) TG_MAP_PTR(name, type), const TG_MAP_PTR(name, type)

#define TG_MAP_STRUCT(name, type) TG_CAT(DynamicArray, type) : TG_TYPE_SPECIFIC_FUNCTION(name, type)
#define TG_MAP_STRUCT_PTR(name, type)                                                              \
    TG_CAT(DynamicArray, type) * : TG_TYPE_SPECIFIC_FUNCTION(name, type)
#define TG_MAP_STRUCT_CONST_PTR(name, type)                                                        \
    TG_MAP_STRUCT_PTR(name, type), const TG_MAP_STRUCT_PTR(name, type)

#ifndef TG_DTYPE0
#define TG_DTYPE0 size_t
#endif
// #define TG_DTYPE0
// #define TG_DTYPE1
// #define TG_DTYPE2
// #define TG_DTYPE3
// #define TG_DTYPE4
// #define TG_DTYPE5
// #define TG_DTYPE6
// #define TG_DTYPE7

#if defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2) && defined(TG_DTYPE3) &&        \
    defined(TG_DTYPE4) && defined(TG_DTYPE5) && defined(TG_DTYPE6) && defined(TG_DTYPE7)
#define TG_MAPS(name, map)                                                                         \
    map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2), map(name, TG_DTYPE3),        \
        map(name, TG_DTYPE4), map(name, TG_DTYPE5), map(name, TG_DTYPE6), map(name, TG_DTYPE7)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2) && defined(TG_DTYPE3) &&      \
    defined(TG_DTYPE4) && defined(TG_DTYPE5) && defined(TG_DTYPE6)
#define TG_MAPS(name, map)                                                                         \
    map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2), map(name, TG_DTYPE3),        \
        map(name, TG_DTYPE4), map(name, TG_DTYPE5), map(name, TG_DTYPE6)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2) && defined(TG_DTYPE3) &&      \
    defined(TG_DTYPE4) && defined(TG_DTYPE5)
#define TG_MAPS(name, map)                                                                         \
    map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2), map(name, TG_DTYPE3),        \
        map(name, TG_DTYPE4), map(name, TG_DTYPE5)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2) && defined(TG_DTYPE3) &&      \
    defined(TG_DTYPE4)
#define TG_MAPS(name, map)                                                                         \
    map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2), map(name, TG_DTYPE3),        \
        map(name, TG_DTYPE4)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2) && defined(TG_DTYPE3)
#define TG_MAPS(name, map)                                                                         \
    map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2), map(name, TG_DTYPE3)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1) && defined(TG_DTYPE2)
#define TG_MAPS(name, map) map(name, TG_DTYPE0), map(name, TG_DTYPE1), map(name, TG_DTYPE2)
#elif defined(TG_DTYPE0) && defined(TG_DTYPE1)
#define TG_MAPS(name, map) map(name, TG_DTYPE0), map(name, TG_DTYPE1)
#elif defined(TG_DTYPE0)
#define TG_MAPS(name, map) map(name, TG_DTYPE0)
#endif

// Type-specific definitions
#ifdef TG_DTYPE0
#define TG_DTYPE TG_DTYPE0
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE1
#define TG_DTYPE TG_DTYPE1
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE2
#define TG_DTYPE TG_DTYPE2
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE3
#define TG_DTYPE TG_DTYPE3
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE4
#define TG_DTYPE TG_DTYPE4
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE5
#define TG_DTYPE TG_DTYPE5
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE6
#define TG_DTYPE TG_DTYPE6
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

#ifdef TG_DTYPE7
#define TG_DTYPE TG_DTYPE7
#include "tgarray_prototype.h"
#undef TG_DTYPE
#endif

// Declare generics
#define array_append(element, array)                                                               \
    _Generic((array), TG_MAPS(array_append, TG_MAP_STRUCT_PTR))(element, array)
#define array_remove(index, array)                                                                 \
    _Generic((array), TG_MAPS(array_remove, TG_MAP_STRUCT_PTR))(index, array)
#define array_get(array, index)                                                                    \
    _Generic((array), TG_MAPS(array_get, TG_MAP_STRUCT_CONST_PTR))(array, index)
// #define array_print(label, array)                                                                  \
//     _Generic((array), TG_MAPS(array_print, TG_MAP_STRUCT_CONST_PTR))(label, array)
#define array_destroy(array) _Generic((array), TG_MAPS(array_destroy, TG_MAP_STRUCT_PTR))(array)
