#pragma once
#include <stddef.h>
#include <stdint.h>

size_t prod(const size_t count, const size_t* arr);

void factorize(const size_t n_initial, size_t* nfactors, size_t* factors);

/** Requires that array is ordered */
size_t unique(const size_t count, size_t* arr);

void transpose(const size_t* in, const size_t nrows, const size_t ncols, size_t* out);

void contract(const size_t* in, const size_t length, const size_t factor, size_t* out);

int64_t mod(const int64_t a, const int64_t b);

void mod_pointwise(const size_t count, const int64_t* a, const int64_t* b, int64_t* c);

void to_spatial(const size_t index, const size_t ndims, const size_t shape[], size_t output[]);

size_t to_linear(const size_t ndims, const size_t index[], const size_t shape[]);

void reverse(const size_t count, size_t arr[]);

void copy(const size_t count, const size_t in[], size_t out[]);

void set(const size_t value, const size_t count, size_t arr[]);