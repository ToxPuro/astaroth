#pragma once
#include <stddef.h>

void matrix_get_row(const size_t row, const size_t nrows, const size_t ncols,
                    const size_t matrix[nrows][ncols], size_t cols[ncols]);

void matrix_remove_row(const size_t row, const size_t nrows, const size_t ncols,
                       const size_t matrix[nrows][ncols], size_t out[nrows - 1][ncols]);

void print_matrix(const char* label, const size_t nrows, const size_t ncols,
                  const size_t matrix[nrows][ncols]);

void test_matrix(void);
