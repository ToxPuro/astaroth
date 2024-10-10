#pragma once
#include <stddef.h>

typedef struct {
    size_t count;
    double* data;
} Buffer;

Buffer bufferCreate(const size_t count);

void bufferDestroy(Buffer* buffer);
