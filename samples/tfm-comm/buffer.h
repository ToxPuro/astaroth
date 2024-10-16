#pragma once
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool on_device;
    size_t count;
    double* data;
} AcBuffer;

AcBuffer acBufferCreate(const size_t count, const bool on_device);

void acBufferDestroy(AcBuffer* buffer);

void acBufferMigrate(const AcBuffer in, AcBuffer* out);

void acBufferPrint(const char* label, const AcBuffer buffer);

#ifdef __cplusplus
}
#endif
