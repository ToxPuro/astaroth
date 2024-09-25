#pragma once
#include <stddef.h>

typedef double BUFFERDATATYPE;

typedef struct {
    size_t length;
    BUFFERDATATYPE* data;
} Buffer;

Buffer acBufferCreate(const size_t length);

void acBufferPrint(const char* label, const Buffer buffer);

void acBufferDestroy(Buffer* buffer);
