#pragma once
#include <stddef.h>

typedef size_t datatype;

typedef struct {
    size_t len;
    size_t capacity;
    datatype* data;
} DynamicArray;

DynamicArray array_create(const size_t capacity);

void array_append(const datatype element, DynamicArray* array);

void array_append_multiple(const size_t count, const datatype* elements, DynamicArray* array);

void array_destroy(DynamicArray* array);
