#pragma once
#include <stdbool.h>
#include <stddef.h>

typedef double real;

#ifdef __cplusplus
extern "C" {
#endif

real* array_create(const size_t count, const bool on_device);

void array_destroy(real** array, const bool on_device);

#ifdef __cplusplus
}
#endif
