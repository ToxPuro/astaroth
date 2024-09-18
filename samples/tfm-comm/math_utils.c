#include "math_utils.h"

#include <string.h> // memmove

#include "errchk.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

/** Product */
size_t
prod(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

static void
test_prod(void)
{
    {
        const size_t arr[] = {1, 2, 3, 4, 5};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(prod(count, arr) == 120);
    }
    {
        const size_t arr[] = {0};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(prod(count, arr) == 0);
    }
}

/** Cumulative product */
void
cumprod(const size_t count, const size_t* restrict in, size_t* restrict out)
{
    ERRCHK(count > 0);

    // Disallow aliasing
    ERRCHK(!(in >= out && in < out + count));
    ERRCHK(!(in + count >= out && in + count < out + count));

    out[0] = in[0];
    for (size_t i = 1; i < count; ++i)
        out[i] = in[i] * out[i - 1];
}

static void
test_cumprod(void)
{
    {
        const size_t a[]   = {0};
        const size_t count = ARRAY_SIZE(a);
        size_t b[count];
        cumprod(count, a, b);
        const size_t c[] = {0};
        ERRCHK(equals(count, b, c));
    }
    {
        const size_t a[]   = {2, 2, 2, 2};
        const size_t count = ARRAY_SIZE(a);
        size_t b[count];
        cumprod(count, a, b);
        const size_t c[] = {2, 4, 8, 16};
        ERRCHK(equals(count, b, c));
    }
    {
        const size_t a[]   = {2, 4, 8, 16};
        const size_t count = ARRAY_SIZE(a);
        size_t b[count];
        cumprod(count, a, b);
        const size_t c[] = {2, 2 * 4, 2 * 4 * 8, 2 * 4 * 8 * 16};
        ERRCHK(equals(count, b, c));
    }
}

size_t
binomial_coefficient(const size_t n, const size_t k)
{
    ERRCHK(n >= k);
    size_t numerator = 1;
    for (size_t i = n; i > n - k; --i)
        numerator *= i;
    size_t denominator = 1;
    for (size_t i = k; i > 0; --i)
        denominator *= i;
    return numerator / denominator;
}

size_t
count_combinations(const size_t n)
{
    size_t result = 0;
    for (size_t k = 0; k <= n; ++k)
        result += binomial_coefficient(n, k);

    return result;
}

static void
test_binomial_coefficient(void)
{
    ERRCHK(binomial_coefficient(52, 5) == 2598960);
    ERRCHK(binomial_coefficient(3, 1) == 3);
    ERRCHK(binomial_coefficient(5, 3) == 10);

    ERRCHK(count_combinations(1) == 2);
    ERRCHK(count_combinations(2) == 4);
    ERRCHK(count_combinations(3) == 8);
}

/** Shift array forward (right) and fill the remaining values.
 * e.g., {1,2,3} -> {fill_value, 1, 2}
 */
void
rshift(const size_t shift, const size_t fill_value, const size_t count, size_t* arr)
{
    ERRCHK(shift < count);
    memmove(&arr[shift], &arr[0], sizeof(arr[0]) * (count - shift));
    set(fill_value, shift, arr);
}

void
test_rshift(void)
{
    {
        size_t a[]         = {1};
        const size_t count = ARRAY_SIZE(a);
        rshift(0, 0, count, a);
        const size_t b[] = {1};
        ERRCHK(equals(count, a, b));
    }
    {
        size_t a[]         = {1, 2, 3, 4};
        const size_t count = ARRAY_SIZE(a);
        rshift(1, 0, count, a);
        const size_t b[] = {0, 1, 2, 3};
        ERRCHK(equals(count, a, b));
    }
    {
        size_t a[]         = {1, 2, 3, 4};
        const size_t count = ARRAY_SIZE(a);
        rshift(3, 5, count, a);
        const size_t b[] = {5, 5, 5, 1};
        ERRCHK(equals(count, a, b));
    }
}

size_t
dot(const size_t count, const size_t* a, const size_t* b)
{
    size_t res = 0;
    for (size_t i = 0; i < count; ++i)
        res += a[i] * b[i];
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

/** Computes the Hamming weight or population count of an array */
size_t
popcount(const size_t count, const size_t* arr)
{
    size_t popcount = 0;
    for (size_t i = 0; i < count; ++i)
        if (arr[i] > 0)
            ++popcount;
    return popcount;
}

static void
test_popcount(void)
{
    {
        const size_t arr[] = {0, 0, 0};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(popcount(count, arr) == 0);
    }

    {
        const size_t arr[] = {0, 1, 0};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(popcount(count, arr) == 1);
    }
    {
        const size_t arr[] = {0, 500, 123};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(popcount(count, arr) == 2);
    }
}

/** Requires that array is ordered
 * Modifies `size_t *arr` inplace to hold only unique values
 * and returns the number of unique values (or the new count) of arr.
 */
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
to_spatial(const size_t index, const size_t ndims, const size_t* shape, size_t* output)
{
    for (size_t j = 0; j < ndims; ++j) {
        size_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        output[j] = (index / divisor) % shape[j];
    }
}

size_t
to_linear(const size_t ndims, const size_t* index, const size_t* shape)
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

void
reverse(const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count / 2; ++i) {
        const size_t tmp   = arr[i];
        arr[i]             = arr[count - i - 1];
        arr[count - i - 1] = tmp;
    }
}

void
copy(const size_t count, const size_t* in, size_t* out)
{
    for (size_t i = 0; i < count; ++i)
        out[i] = in[i];
}

void
set(const size_t value, const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count; ++i)
        arr[i] = value;
}

void
iset(const int value, const size_t count, int* arr)
{
    for (size_t i = 0; i < count; ++i)
        arr[i] = value;
}

void
add_to_array(const size_t value, const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count; ++i)
        arr[i] += value;
}

void
add_arrays(const size_t count, const size_t* a, const size_t* b, size_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = a[i] + b[i];
}

// Autogenerated
int
min_int(const int a, const int b)
{
    return a < b ? a : b;
}
int64_t
min_int64_t(const int64_t a, const int64_t b)
{
    return a < b ? a : b;
}
size_t
min_size_t(const size_t a, const size_t b)
{
    return a < b ? a : b;
}
float
min_float(const float a, const float b)
{
    return a < b ? a : b;
}
double
min_double(const double a, const double b)
{
    return a < b ? a : b;
}

/*
 * Unit testing
 */
bool
equals(const size_t count, const size_t* a, const size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

void
test_math_utils(void)
{
    test_prod();
    test_cumprod();
    test_rshift();
    test_popcount();
    test_binomial_coefficient();
}
