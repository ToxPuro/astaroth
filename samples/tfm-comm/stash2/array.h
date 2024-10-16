#pragma once
#include <stddef.h>

typedef struct {
    size_t length;
    size_t capacity;
    size_t element_size;
    char* data;
} Array;

Array arrayCreate(const size_t element_size, const size_t initial_capacity);

void arrayAppend(const void* element, Array* array);

char* arrayGet(const Array array, const size_t index);

void arrayRemove(const size_t index, Array* array);

void arrayDestroy(Array* array);
