#include "matrix.h"

#include "ndarray.h"

void
print_matrix(const char* label, const size_t nrows, const size_t ncols,
             const size_t matrix[nrows][ncols])
{
    print_ndarray(label, 2, (size_t[]){ncols, nrows}, (size_t*)matrix);
}
