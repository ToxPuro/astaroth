#include "math_utils.h"

#include "errchk.h"

size_t
prod(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

void
factorize(const size_t n_initial, size_t* nfactors, size_t* factors)
{
    ERRCHK(nfactors);
    size_t n     = n_initial;
    size_t count = 0;
    if (factors == NULL) {
        for (size_t i = 2; i <= n; ++i)
            while ((n % i) == 0) {
                ++count;
                n /= i;
            }
    }
    else {
        for (size_t i = 2; i <= n; ++i)
            while ((n % i) == 0) {
                factors[count++] = i;
                n /= i;
            }
    }
    *nfactors = count;
}

/** Requires that array is ordered */
size_t
unique(const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count - 1; ++i)
        ERRCHK(arr[i + 1] >= arr[i]);

    ERRCHK(count > 0);
    size_t num_unique = 0;
    for (size_t i = 0; i < count; ++i) {
        arr[num_unique] = arr[i];
        ++num_unique;
        while ((i + 1 < count) && (arr[i + 1] == arr[i]))
            ++i;
    }

    return num_unique;
}

void
transpose(const size_t* in, const size_t nrows, const size_t ncols, size_t* out)
{
    for (size_t i = 0; i < ncols; ++i) {
        for (size_t j = 0; j < nrows; ++j) {
            out[j + i * nrows] = in[i + j * ncols];
        }
    }
}

void
contract(const size_t* in, const size_t length, const size_t factor, size_t* out)
{
    ERRCHK((length % factor) == 0);
    const size_t out_length = length / factor;
    for (size_t j = 0; j < out_length; ++j) {
        out[j] = 1;
        for (size_t i = 0; i < factor; ++i)
            out[j] *= in[i + j * factor];
    }
}

int64_t
mod(const int64_t a, const int64_t b)
{
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

void
mod_pointwise(const size_t count, const int64_t* a, const int64_t* b, int64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = mod(a[i], b[i]);
}

void
to_spatial(const size_t index, const size_t ndims, const size_t shape[], size_t output[])
{
    for (size_t j = 0; j < ndims; ++j) {
        size_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        output[j] = (index / divisor) % shape[j];
    }
}

size_t
to_linear(const size_t ndims, const size_t index[], const size_t shape[])
{
    size_t result = 0;
    for (size_t j = 0; j < ndims; ++j) {
        size_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += index[j] * factor;
    }
    return result;
}