#pragma once
#include <stddef.h>

typedef struct {
    size_t count;
    double* data;
} Buffer;

#ifdef __cplusplus
extern "C" {
#endif

Buffer buffer_create(const size_t count);

void buffer_destroy(Buffer* buffer);

#ifdef __cplusplus
}
#endif
