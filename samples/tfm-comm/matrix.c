#include "matrix.h"

#include <string.h>

#include "array.h"
#include "errchk.h"
#include "math_utils.h" // equals
#include "ndarray.h"
#include "print.h"

void
matrix_get_row(const size_t row, const size_t nrows, const size_t ncols,
               const size_t matrix[nrows][ncols], size_t cols[ncols])
{
    ERRCHK(row < nrows);
    copy(ncols, matrix[row], cols);
}

void
test_get_row(void)
{
    {
        const size_t nrows            = 5;
        const size_t ncols            = 2;
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {7, 8};
        size_t out[ncols];
        matrix_get_row(3, nrows, ncols, in, out);
        ERRCHK(equals(ncols, row, out));
    }
    {
        const size_t nrows            = 5;
        const size_t ncols            = 2;
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {1, 2};
        size_t out[ncols];
        matrix_get_row(0, nrows, ncols, in, out);
        ERRCHK(equals(ncols, row, out));
    }
    {
        const size_t nrows            = 5;
        const size_t ncols            = 2;
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {9, 10};
        size_t out[ncols];
        matrix_get_row(nrows - 1, nrows, ncols, in, out);
        ERRCHK(equals(ncols, row, out));
    }
}

void
matrix_remove_row(const size_t row, const size_t nrows, const size_t ncols,
                  const size_t in[nrows][ncols], size_t out[nrows - 1][ncols])
{
    ERRCHK(row < nrows);

    memmove(out, in, row * sizeof(in[0]));

    const size_t count = nrows - row - 1;
    if (count > 0)
        memmove(out[row], in[row + 1], count * sizeof(in[0]));
}

void
test_remove_row(void)
{
    {
        const size_t nrows            = 5;
        const size_t ncols            = 2;
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t model[nrows - 1][ncols] = {
            {1, 2},
            {3, 4},
            {7, 8},
            {9, 10},
        };
        size_t out[nrows - 1][ncols];
        matrix_remove_row(2, nrows, ncols, in, out);
        ERRCHK(equals((nrows - 1) * ncols, (size_t*)model, (size_t*)out));
    }
    {
        const size_t nrows            = 5;
        const size_t ncols            = 2;
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t model[nrows - 1][ncols] = {
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10},
        };
        size_t out[nrows - 1][ncols];
        matrix_remove_row(0, nrows, ncols, in, out);
        ERRCHK(equals((nrows - 1) * ncols, (size_t*)model, (size_t*)out));
    }
    {
        const size_t nrows                   = 5;
        const size_t ncols                   = 2;
        const size_t in[nrows][ncols]        = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
        const size_t model[nrows - 1][ncols] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        size_t out[nrows - 1][ncols];
        matrix_remove_row(nrows - 1, nrows, ncols, in, out);
        ERRCHK(equals((nrows - 1) * ncols, (size_t*)model, (size_t*)out));
    }
}

void
print_matrix(const char* label, const size_t nrows, const size_t ncols,
             const size_t matrix[nrows][ncols])
{
    print_ndarray(label, 2, (size_t[]){ncols, nrows}, (size_t*)matrix);
}

void
test_matrix(void)
{
    test_get_row();
    test_remove_row();
}
