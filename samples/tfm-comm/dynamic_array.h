#pragma once
#include <stddef.h>

typedef size_t DATYPE;

typedef struct {
    size_t len;
    size_t capacity;
    DATYPE* data;
} DynamicArray;

DynamicArray array_create(const size_t capacity);

void array_append(const DATYPE element, DynamicArray* array);

void array_append_multiple(const size_t count, const DATYPE* elements, DynamicArray* array);

void array_destroy(DynamicArray* array);

// typedef struct {
//     size_t ndims;
//     size_t* shape;
// } Shape;

// typedef struct {
//     DynamicArray array;
//     Shape shape;
// } NdArray;
