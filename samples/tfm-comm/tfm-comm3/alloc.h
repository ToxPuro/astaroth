#pragma once
#include <stdbool.h>
#include <stddef.h>

/**
 * Allocates bytes of uninitialized memory
 */
void* ac_malloc(const size_t bytes);

/**
 * Allocate zero-initialized memory for `count` elements of size `size`
 */
void* ac_calloc(const size_t count, const size_t size);

/**
 * Free allocated memory
 */
void ac_free(void* ptr);

/**
 * Reallocate a memory segment to hold `count` elements of size `size`
 */
void* ac_realloc(const size_t count, const size_t size, void* ptr);

/**
 * Copy n elements from memory buffer `in` to `out`
 */
void ac_copy(const size_t count, const size_t size, const void* in, void* out);

/**
 * Allocate and duplicate a memory segment
 */
void* ac_dup(const size_t count, const size_t size, const void* ptr);

/**
 * Compare two blocks of memory.
 * The arrays must have the same length and element size.
 */
bool ac_cmp(const size_t count, const size_t size, const void* a, const void* b);
