#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool any(const size_t count, const bool* arr);

bool all(const size_t count, const bool* arr);

uint64_t prod(const size_t count, const uint64_t* arr);

void cumprod(const size_t count, const uint64_t* restrict in, uint64_t* restrict out);

uint64_t powzu(const uint64_t base, const uint64_t exponent);

uint64_t binomial_coefficient(const uint64_t n, const uint64_t k);

size_t count_combinations(const size_t n);

void rshift(const uint64_t shift, const uint64_t fill_value, const size_t count, uint64_t* arr);

uint64_t dot(const size_t count, const uint64_t* a, const uint64_t* b);

void factorize(const size_t n_initial, size_t* nfactors, uint64_t* factors);

uint64_t popcount(const size_t count, const uint64_t* arr);

void arange(const uint64_t start_value, const size_t count, uint64_t* arr);

/** Requires that array is ordered */
uint64_t unique(const size_t count, uint64_t* arr);

uint64_t unique_subsets(const size_t count, const uint64_t* a, uint64_t subset_length, uint64_t* b);

void transpose(const uint64_t* in, const uint64_t nrows, const uint64_t ncols, uint64_t* out);

void contract(const uint64_t* in, const size_t length, const uint64_t factor, uint64_t* out);

int64_t mod(const int64_t a, const int64_t b);

void mod_pointwise(const size_t count, const int64_t* a, const int64_t* b, int64_t* c);

/** Calculates a projection from a linear index to spatial coordinates within the shape */
void to_spatial(const uint64_t index, const size_t ndims, const uint64_t* shape, uint64_t* coords);

/** Calculates a projection from spatial coordinates within shape to a linear index */
uint64_t to_linear(const size_t ndims, const uint64_t* coords, const uint64_t* shape);

void reverse(const size_t count, uint64_t* arr);

void reversei(const size_t count, int* arr);

void set(const uint64_t value, const size_t count, uint64_t* arr);

void set_array_int(const int value, const size_t count, int* arr);

void array_set_uint64_t(const uint64_t value, const size_t count, uint64_t* arr);

void add_to_array(const uint64_t value, const size_t count, uint64_t* arr);

void add(const size_t count, const uint64_t* a, uint64_t* b);

void add_arrays(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);

void subtract_arrays(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);

void subtract_value(const uint64_t value, const size_t count, uint64_t* arr);

/** Calculates the element-wise product (Hadamard product) of two flattened matrices */
void array_mul(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);

void array_div(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);

void array_div_int(const size_t count, const int* a, const int* b, int* c);

/** Repeats `count` elements in `a` `nrepeats` times and writes the result to b.
 * `b`is required to be able to hold at least `count*nrepeats` elements.
 * E.g. repeat(2, (uint64_t[]){1,2}, 3, ...) -> {1, 2, 1, 2, 1, 2}
 */
void repeat(const size_t count, const uint64_t* a, const uint64_t nrepeats, uint64_t* b);

void swap(const uint64_t i, const uint64_t j, const size_t count, uint64_t* arr);

void sort(const size_t count, uint64_t* arr);

bool intersect_lines(const uint64_t a1, const uint64_t a2, const uint64_t b1, const uint64_t b2);

bool intersect_box_note_changed(const size_t ndims, const uint64_t* a_dims,
                                const uint64_t* a_offset, const uint64_t* b_dims,
                                const uint64_t* b_offset);

bool within_box_note_changed(const size_t ndims, const uint64_t* coords, const uint64_t* box_dims,
                             const uint64_t* box_offset);

/**
 * Returns the next positive integer in range [0, INT_MAX].
 * Returns 0 if the counter overflows or is negative.
 */
int next_positive_integer(int counter);

/** Ndarray */
void set_ndarray_uint64_t(const uint64_t value, const size_t ndims, const uint64_t* dims,
                          const uint64_t* subdims, const uint64_t* start, uint64_t* arr);

void set_ndarray_void(const uint64_t element_size, const void* value, const size_t ndims,
                      const uint64_t* dims, const uint64_t* subdims, const uint64_t* start,
                      char* arr);

void set_ndarray_double(const double value, const size_t ndims, const uint64_t* dims,
                        const uint64_t* subdims, const uint64_t* start, double* arr);

/** Checks whether all of the `count` elements starting from start_a and start_b are equal */
bool ndarray_equals(const size_t count, const size_t ndims, const uint64_t* a_offset,
                    const uint64_t* b_offset, const uint64_t* dims, const uint64_t* arr);

/** Matrix */
void matrix_get_row(const uint64_t row, const uint64_t nrows, const uint64_t ncols,
                    const uint64_t* mat, uint64_t* cols);

void matrix_remove_row(const uint64_t row, const uint64_t nrows, const uint64_t ncols,
                       const uint64_t* in, uint64_t* out);

bool matrix_row_equals(const uint64_t row, const uint64_t nrows, const uint64_t ncols,
                       const uint64_t* mat, const uint64_t* cols);

/** Unit testing */
bool equals(const size_t count, const uint64_t* a, const uint64_t* b);
bool all_less_than(const size_t count, const uint64_t* a, const uint64_t* b);
bool all_less_or_equal_than(const size_t count, const uint64_t* a, const uint64_t* b);
void test_math_utils(void);
