#pragma once
#include <stddef.h>

// create
// destroy
// push
// pop
// length
// get

typedef struct vec_s Vector;

Vector* vector_create(const size_t capacity);

void vector_destroy(Vector* vec);

void vector_push(const void* ptr, Vector* vec);

void* vector_pop(Vector* vec);

size_t vector_len(const Vector* vec);

void* vector_get(const Vector* vec, const size_t i);

void vector_test(void);
