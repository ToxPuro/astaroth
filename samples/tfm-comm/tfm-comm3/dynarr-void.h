#pragma once
#include <stddef.h>

typedef struct {
    size_t len;
    size_t capacity;
    size_t elem_size;
    void* data;
} DynamicArray;

/**
 * Creates a dynamic array with `ca` elements of size `size`
 */
DynamicArray dynarr_create(const size_t capacity, const size_t elem_size);

void dynarr_destroy(DynamicArray* arr);

void dynarr_append(const size_t elem_size, const void* elem, DynamicArray* arr);

void dynarr_append_multiple(const size_t count, const size_t elem_size, const void* elems,
                            DynamicArray* arr);

void dynarr_remove(const size_t index, DynamicArray* arr);

void dynarr_remove_multiple(const size_t index, const size_t count, DynamicArray* arr);

void* dynarr_get(const size_t index, const DynamicArray arr);

void print_dynarr(const char* label, const DynamicArray arr);

unsigned test_dynarr(void);
