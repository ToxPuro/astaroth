#pragma once
#include <stddef.h>

typedef size_t size_t;

typedef struct {
    size_t length;
    size_t capacity;
    size_t* data;
} DynamicArray;

DynamicArray array_create(const size_t capacity);

void array_append(const size_t element, DynamicArray* array);

void array_append_multiple(const size_t count, const size_t* elements, DynamicArray* array);

void array_remove(const size_t index, DynamicArray* array);

size_t array_get(const DynamicArray array, const size_t index);

void to_static_array(const DynamicArray in, const size_t nrows, const size_t ncols,
                     size_t out[nrows][ncols]);

void array_destroy(DynamicArray* array);
