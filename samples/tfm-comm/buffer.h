#pragma once
#include <stddef.h>

typedef double BUFFERDATATYPE;

typedef struct {
    size_t length;
    BUFFERDATATYPE* data;
} Buffer;

Buffer acBufferCreate(const size_t length);

void acBufferDestroy(Buffer* buffer);
