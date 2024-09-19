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
//     size_t* dims;

//     DynamicArray array;
// } DynamicNdArray;

// DynamicNdArray ndarray_create(const size_t ndims, const size_t* dims);

// void ndarray_append(const size_t ndims, const size_t* row, DynamicNdArray* ndarray);

// void ndarray_print(const char* label, const DynamicNdArray ndarray);

// void ndarray_destroy(DynamicNdArray* ndarray);
