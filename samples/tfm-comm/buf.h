#pragma once
#include <stddef.h>

typedef struct {
    size_t count;
    double* data;
} Buffer;

Buffer buffer_create(const size_t count);

void buffer_destroy(Buffer* buffer);
