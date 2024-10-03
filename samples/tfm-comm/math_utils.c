#include "math_utils.h"

#include <limits.h> // INT_MAX

#include "errchk.h"
#include "nalloc.h"
#include "print.h"

#include "misc.h"

bool
any(const size_t count, const bool* arr)
{
    bool res = false;
    for (size_t i = 0; i < count; ++i)
        res = res || arr[i];
    return res;
}

static void
test_any(void)
{
    {
        const bool arr[]   = {false, false, false};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(any(count, arr) == false);
    }
    {
        const bool arr[]   = {false, true, false};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(any(count, arr) == true);
    }
    {
        const bool arr[] = {true};
        ERRCHK(any(0, arr) == false);
    }
}

bool
all(const size_t count, const bool* arr)
{
    bool res = true;
    for (size_t i = 0; i < count; ++i)
        res = res && arr[i];
    return res;
}

static void
test_all(void)
{
    {
        const bool arr[]   = {false, false, false};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(all(count, arr) == false);
    }
    {
        const bool arr[]   = {true, false, true};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(all(count, arr) == false);
    }
    {
        const bool arr[]   = {true, true, true};
        const size_t count = ARRAY_SIZE(arr);
        ERRCHK(all(count, arr) == true);
    }
    {
        const bool arr[] = {true};
        ERRCHK(all(0, arr) == true);
    }
}

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

size_t
powzu(const size_t base, const size_t exponent)
{
    if (exponent == 0) {
        return 1;
    }
    else if (base <= 1) {
        return base;
    }
    else {
        size_t res = 1;
        for (size_t i = 0; i < exponent; ++i) {
            ERRCHK(base <= SIZE_MAX / res); // Overflow
            res *= base;
        }
        return res;
    }
}

static void
test_powzu(void)
{
    ERRCHK(powzu(0, 0) == 1);
    ERRCHK(powzu(0, 1) == 0);
    ERRCHK(powzu(0, 123456) == 0);
    ERRCHK(powzu(1, 0) == 1);
    ERRCHK(powzu(1, 1) == 1);
    ERRCHK(powzu(1, 123456) == 1);
    ERRCHK(powzu(2, 8) == 256);
    ERRCHK(powzu(0, 1) == 0);
    ERRCHK(powzu(128, 0) == 1);
    ERRCHK(powzu(7, 5) == 16807);
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
        size_t* b;
        nalloc(count, b);
        cumprod(count, a, b);
        const size_t c[] = {0};
        ERRCHK(equals(count, b, c));
        ndealloc(b);
    }
    {
        const size_t a[]   = {2, 2, 2, 2};
        const size_t count = ARRAY_SIZE(a);
        size_t* b;
        nalloc(count, b);
        cumprod(count, a, b);
        const size_t c[] = {2, 4, 8, 16};
        ERRCHK(equals(count, b, c));
        ndealloc(b);
    }
    {
        const size_t a[]   = {2, 4, 8, 16};
        const size_t count = ARRAY_SIZE(a);
        size_t* b;
        nalloc(count, b);
        cumprod(count, a, b);
        const size_t c[] = {2, 2 * 4, 2 * 4 * 8, 2 * 4 * 8 * 16};
        ERRCHK(equals(count, b, c));
        ndealloc(b);
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

static void
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

void
arange(const size_t start_value, const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count; ++i)
        arr[i] = start_value + i;
}

/** Requires that array is ordered
 * Modifies `size_t *arr` inplace to hold only unique values
 * and returns the number of unique values (or the new count) of arr.
 */
size_t
unique(const size_t count, size_t* arr)
{
    sort(count, arr);

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

static void
test_unique(void)
{
    srand(12345);
    const size_t nsamples  = 128;
    const size_t max_count = 32;
    for (size_t i = 0; i < nsamples; ++i) {
        const size_t count = ((size_t)rand()) % max_count + 1;

        size_t* arr;
        nalloc(count, arr);
        for (size_t j = 0; j < count; ++j)
            arr[j] = (size_t)rand();

        unique(count, arr);
        for (size_t j = 0; j < count; ++j)
            for (size_t k = j + 1; k < count; ++k)
                ERRCHK(arr[j] != arr[k]);
        ndealloc(arr);
    }
}

static bool
contains_subset(const size_t subset_length, const size_t* subset, const size_t count,
                const size_t* b)
{
    for (size_t i = 0; i < count; i += subset_length) {
        if (equals(subset_length, subset, &b[i]))
            return true;
    }
    return false;
}

size_t
unique_subsets(const size_t count, const size_t* a, size_t subset_length, size_t* b)
{
    ERRCHK(b != NULL);
    size_t nsubsets = 0;
    for (size_t i = 0; i < count; i += subset_length) {
        if (!contains_subset(subset_length, &a[i], subset_length * nsubsets, b)) {
            ncopy(subset_length, &a[i], &b[subset_length * nsubsets]);
            ++nsubsets;
        }
    }
    return subset_length * nsubsets;
}

static void
test_unique_subsets(void)
{
    {
        const size_t a[]           = {1, 2, 3, 4};
        const size_t subset_length = 1;
        const size_t model[]       = {1, 2, 3, 4};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
    {
        const size_t a[]           = {1, 1, 3, 4};
        const size_t subset_length = 1;
        const size_t model[]       = {1, 3, 4};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
    {
        const size_t a[]           = {1, 1, 2, 2, 1, 1};
        const size_t subset_length = 2;
        const size_t model[]       = {1, 1, 2, 2};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
    {
        const size_t a[]           = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3};
        const size_t subset_length = 3;
        const size_t model[]       = {1, 1, 1, 1, 2, 3};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
    {
        const size_t a[]           = {1, 2, 3, 3, 2, 1, 1, 2, 3};
        const size_t subset_length = 3;
        const size_t model[]       = {1, 2, 3, 3, 2, 1};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
    {
        const size_t a[]           = {1, 2, 3, 3, 2, 1, 1, 3, 2, 3, 2, 1};
        const size_t subset_length = 3;
        const size_t model[]       = {1, 2, 3, 3, 2, 1, 1, 3, 2};

        const size_t count              = ARRAY_SIZE(a);
        const size_t output_count_model = ARRAY_SIZE(model);
        size_t* b;
        nalloc(count, b);
        const size_t output_count = unique_subsets(count, a, subset_length, b);
        ERRCHK(output_count == output_count_model);
        ERRCHK(equals(output_count, b, model));
        ndealloc(b);
    }
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

    // Alternative
    // size_t* basis;
    // nalloc(ndims, basis);
    // cumprod(ndims, shape, basis);
    // rshift(1, 1, ndims, basis);

    // for (size_t i = 0; i < ndims; ++i)
    //     output[i] = (index / basis[i]) % shape[i];

    // ndealloc(basis);
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

    // Alternative
    // size_t* basis;
    // nalloc(ndims, basis);
    // cumprod(ndims, shape, basis);
    // rshift(1, 1, ndims, basis);
    // const size_t result = dot(ndims, index, basis);
    // ndealloc(basis);
    // return result;
}

static void
test_to_spatial_to_linear(void)
{
    {

        const size_t index          = 0;
        const size_t shape[]        = {8, 8, 8};
        const size_t ndims          = ARRAY_SIZE(shape);
        const size_t model_coords[] = {0, 0, 0};

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t index          = 1;
        const size_t shape[]        = {8, 8, 8};
        const size_t ndims          = ARRAY_SIZE(shape);
        const size_t model_coords[] = {1, 0, 0};

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t index          = 8;
        const size_t shape[]        = {8, 8, 8};
        const size_t ndims          = ARRAY_SIZE(shape);
        const size_t model_coords[] = {0, 1, 0};

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t index          = 8 * 8;
        const size_t shape[]        = {8, 8, 8};
        const size_t ndims          = ARRAY_SIZE(shape);
        const size_t model_coords[] = {0, 0, 1};

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t shape[]        = {5, 7, 9};
        const size_t model_coords[] = {1, 2, 3};
        const size_t index          = model_coords[0] + model_coords[1] * shape[0] +
                             model_coords[2] * shape[0] * shape[1];
        const size_t ndims = ARRAY_SIZE(shape);

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t shape[]        = {4701, 12, 525};
        const size_t model_coords[] = {591, 5, 255};
        const size_t index          = model_coords[0] + model_coords[1] * shape[0] +
                             model_coords[2] * shape[0] * shape[1];
        const size_t ndims = ARRAY_SIZE(shape);

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t shape[]        = {500, 250};
        const size_t model_coords[] = {499, 249};
        const size_t index          = model_coords[0] + model_coords[1] * shape[0];
        const size_t ndims          = ARRAY_SIZE(shape);

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {

        const size_t shape[]        = {2, 2};
        const size_t model_coords[] = {1, 0};
        const size_t index          = 1;
        const size_t ndims          = ARRAY_SIZE(shape);

        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(index, ndims, shape, coords);
        ERRCHK(equals(ndims, coords, model_coords));
        ERRCHK(to_linear(ndims, coords, shape) == index);
        ndealloc(coords);
    }
    {
        const size_t ndims = 2;
        size_t* coords;
        nalloc(ndims, coords);
        to_spatial(1, ndims, (size_t[]){2, 2}, coords);
        ERRCHK(equals(ndims, coords, (size_t[]){1, 0}));
        ndealloc(coords);
    }
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
reversei(const size_t count, int* arr)
{
    for (size_t i = 0; i < count / 2; ++i) {
        const int tmp      = arr[i];
        arr[i]             = arr[count - i - 1];
        arr[count - i - 1] = tmp;
    }
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
add(const size_t count, const size_t* a, size_t* b)
{
    // Disallow aliasing
    ERRCHK(!(a >= b && a < b + count));
    ERRCHK(!(a + count >= b && a + count < b + count));

    for (size_t i = 0; i < count; ++i)
        b[i] += a[i];
}

void
add_arrays(const size_t count, const size_t* a, const size_t* b, size_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = a[i] + b[i];
}

void
subtract_arrays(const size_t count, const size_t* a, const size_t* b, size_t* c)
{
    for (size_t i = 0; i < count; ++i) {
        ERRCHK(a[i] >= b[i]);
        c[i] = a[i] - b[i];
    }
}

void
subtract_value(const size_t value, const size_t count, size_t* arr)
{
    for (size_t i = 0; i < count; ++i) {
        ERRCHK(arr[i] >= value);
        arr[i] -= value;
    }
}

void
mul(const size_t count, const size_t* a, const size_t* b, size_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = a[i] * b[i];
}

void
repeat(const size_t count, const size_t* a, const size_t nrepeats, size_t* b)
{
    for (size_t i = 0; i < nrepeats; ++i)
        ncopy(count, a, &b[i * count]);
}

static void
test_repeat(void)
{
    {
        const size_t a[]      = {1, 2};
        const size_t nrepeats = 3;
        const size_t model[]  = {1, 2, 1, 2, 1, 2};

        const size_t count = ARRAY_SIZE(a);
        size_t* b;
        nalloc(count * nrepeats, b);
        repeat(count, a, nrepeats, b);
        ERRCHK(equals(count * nrepeats, b, model) == true);
        ndealloc(b);
    }
    {
        const size_t a[]      = {1};
        const size_t nrepeats = 3;
        const size_t model[]  = {1, 1, 1};

        const size_t count = ARRAY_SIZE(a);
        size_t* b;
        nalloc(count * nrepeats, b);
        repeat(count, a, nrepeats, b);
        ERRCHK(equals(count * nrepeats, b, model) == true);
        ndealloc(b);
    }
    {
        const size_t a[]      = {1, 2, 3, 4};
        const size_t nrepeats = 2;
        const size_t model[]  = {1, 2, 3, 4, 1, 2, 3, 4};

        const size_t count = ARRAY_SIZE(a);
        size_t* b;
        nalloc(count * nrepeats, b);
        repeat(count, a, nrepeats, b);
        ERRCHK(equals(count * nrepeats, b, model) == true);
        ndealloc(b);
    }
}

void
swap(const size_t i, const size_t j, const size_t count, size_t* arr)
{
    ERRCHK(i < count);
    ERRCHK(j < count);
    const size_t tmp = arr[i];
    arr[i]           = arr[j];
    arr[j]           = tmp;
}

void
sort(const size_t count, size_t* arr)
{
    ERRCHK(count > 0);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            if (arr[j] < arr[i])
                swap(i, j, count, arr);
        }
    }
}

static void
test_sort(void)
{
    srand(12345);
    const size_t nsamples  = 128;
    const size_t max_count = 32;
    for (size_t i = 0; i < nsamples; ++i) {
        const size_t count = ((size_t)rand()) % max_count + 1;

        size_t* arr;
        nalloc(count, arr);
        for (size_t j = 0; j < count; ++j)
            arr[j] = (size_t)rand();
        sort(count, arr);
        for (size_t j = 1; j < count; ++j)
            ERRCHK(arr[j] >= arr[j - 1]);
        ndealloc(arr);
    }
}

/** Returns true if the lines on intervals [a1, a2) and [b1, b2) intersect */
bool
intersect_lines(const size_t a1, const size_t a2, const size_t b1, const size_t b2)
{
    return (a1 >= b1 && a1 < b2) || (b1 >= a1 && b1 < a2);
}

static void
test_intersect_lines(void)
{
    ERRCHK(intersect_lines(0, 1, 1, 2) == false);
    ERRCHK(intersect_lines(0, 3, 1, 1) == true);
    ERRCHK(intersect_lines(0, 3, 2, 3) == true);
    ERRCHK(intersect_lines(1, 2, 0, 1) == false);
    ERRCHK(intersect_lines(1, 2, 0, 2) == true);
    ERRCHK(intersect_lines(1, 3, 0, 4) == true);
    ERRCHK(intersect_lines(0, 4, 1, 3) == true);
}

bool
intersect_box(const size_t ndims, const size_t* a_start, const size_t* a_dims,
              const size_t* b_start, const size_t* b_dims)
{
    bool all_intersect = true;
    for (size_t i = 0; i < ndims; ++i)
        all_intersect = all_intersect && intersect_lines(a_start[i], a_start[i] + a_dims[i],
                                                         b_start[i], b_start[i] + b_dims[i]);

    return all_intersect;
}

/** Check if coords are within the box spanned by box_min (inclusive) and box_max (exclusive) */
bool
within_box(const size_t ndims, const size_t* coords, const size_t* box_min, const size_t* box_max)
{
    bool res = true;
    for (size_t i = 0; i < ndims; ++i)
        res = res && coords[i] >= box_min[i] && coords[i] < box_max[i];
    return res;
}

static void
test_within_box(void)
{
    {
        const size_t box_min[] = {0, 0, 0};
        const size_t box_max[] = {10, 10, 10};
        const size_t coords[]  = {0, 0, 0};
        const size_t ndims     = ARRAY_SIZE(coords);
        ERRCHK(within_box(ndims, coords, box_min, box_max) == true);
    }
    {
        const size_t box_min[] = {0, 0, 0};
        const size_t box_max[] = {10, 10, 10};
        const size_t coords[]  = {0, 10, 0};
        const size_t ndims     = ARRAY_SIZE(coords);
        ERRCHK(within_box(ndims, coords, box_min, box_max) == false);
    }
    {
        const size_t box_min[] = {0, 0, 0};
        const size_t box_max[] = {10, 10, 10};
        const size_t coords[]  = {11, 11, 11};
        const size_t ndims     = ARRAY_SIZE(coords);
        ERRCHK(within_box(ndims, coords, box_min, box_max) == false);
    }
    {
        const size_t box_min[] = {0, 0, 0, 0, 0, 0, 0};
        const size_t box_max[] = {1, 2, 3, 4, 5, 6, 7};
        const size_t coords[]  = {0, 1, 2, 3, 4, 5, 6};
        const size_t ndims     = ARRAY_SIZE(coords);
        ERRCHK(within_box(ndims, coords, box_min, box_max) == true);
    }
}

static void
test_intersect_box(void)
{
    {
        const size_t a[]      = {0, 0, 0};
        const size_t a_dims[] = {1, 1, 1};
        const size_t b[]      = {0, 0, 0};
        const size_t b_dims[] = {1, 1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == true);
    }
    {
        const size_t a[]      = {0, 0};
        const size_t a_dims[] = {1, 1};
        const size_t b[]      = {1, 0};
        const size_t b_dims[] = {1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == false);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == false);
    }
    {
        const size_t a[]      = {0, 0};
        const size_t a_dims[] = {2, 2};
        const size_t b[]      = {1, 2};
        const size_t b_dims[] = {1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == false);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == false);
    }
    {
        const size_t a[]      = {0, 0};
        const size_t a_dims[] = {2, 2};
        const size_t b[]      = {1, 1};
        const size_t b_dims[] = {1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == true);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == true);
    }
    {
        const size_t a[]      = {0, 0, 0, 0, 0};
        const size_t a_dims[] = {2, 2, 2, 2, 2};
        const size_t b[]      = {1, 1, 1, 1, 1};
        const size_t b_dims[] = {1, 1, 1, 1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == true);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == true);
    }
    {
        const size_t a[]      = {0, 0, 0, 0, 0};
        const size_t a_dims[] = {2, 2, 2, 2, 2};
        const size_t b[]      = {1, 1, 2, 1, 1};
        const size_t b_dims[] = {1, 1, 1, 1, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == false);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == false);
    }
    {
        const size_t a[]      = {1, 0};
        const size_t a_dims[] = {1, 3};
        const size_t b[]      = {0, 1};
        const size_t b_dims[] = {3, 1};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == true);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == true);
    }
    {
        const size_t a[]      = {5, 0};
        const size_t a_dims[] = {5, 10};
        const size_t b[]      = {0, 5};
        const size_t b_dims[] = {10, 5};
        const size_t ndims    = ARRAY_SIZE(a);
        ERRCHK(intersect_box(ndims, a, a_dims, b, b_dims) == true);
        ERRCHK(intersect_box(ndims, b, b_dims, a, a_dims) == true);
    }
}

int
next_positive_integer(int counter)
{
    ++counter;
    if (counter < 0)
        counter = 0;
    return counter;
}

static void
test_next_positive_integer(void)
{
    const int nsamples = 4096;
    {
        // Test ordering
        int prev = -500;
        for (size_t i = 0; i < nsamples; ++i) {
            int curr = next_positive_integer(prev);
            if (prev == INT_MAX)
                ERRCHK(curr == 0);
            else
                ERRCHK(curr > prev);
            prev = curr;
        }
    }
    {
        // Test wrap-around
        int prev = INT_MAX - nsamples / 2;
        for (size_t i = 0; i < nsamples; ++i) {
            int curr = next_positive_integer(prev);
            if (prev == INT_MAX)
                ERRCHK(curr == 0);
            else
                ERRCHK(curr > prev);
            prev = curr;
        }
    }
}

/*
 * Ndarray
 */
void
set_ndarray(const size_t value, const size_t ndims, const size_t* start, const size_t* subdims,
            const size_t* dims, size_t* arr)
{
    if (ndims == 0) {
        *arr = value;
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const size_t offset = prod(ndims - 1, dims);
        for (size_t i = start[ndims - 1]; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            set_ndarray(value, ndims - 1, start, subdims, dims, &arr[i * offset]);
    }
}

/** Note: duplicate with to_linear and to_spatial */
static size_t
nd_to_1d(const size_t ndims, const size_t* coords, const size_t* dims)
{
    ERRCHK(all_less_than(ndims, coords, dims));
    size_t* offset;
    nalloc(ndims, offset);
    cumprod(ndims, dims, offset);
    rshift(1, 1, ndims, offset);
    const size_t res = dot(ndims, coords, offset);
    ndealloc(offset);
    return res;
}

static void
test_nd_to_1d(void)
{
    {
        const size_t coords[] = {0, 0, 0};
        const size_t dims[]   = {1, 1, 1};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 0);
    }
    {
        const size_t coords[] = {1, 0};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 1);
    }
    {
        const size_t coords[] = {31, 0};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 31);
    }
    {
        const size_t coords[] = {0, 31};
        const size_t dims[]   = {32, 32};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 31 * 32);
    }
    {
        const size_t coords[] = {1, 2, 3, 4};
        const size_t dims[]   = {10, 9, 8, 7};
        const size_t ndims    = ARRAY_SIZE(dims);
        ERRCHK(nd_to_1d(ndims, coords, dims) == 1 + 2 * 10 + 3 * 10 * 9 + 4 * 10 * 9 * 8);
    }
}

bool
ndarray_equals(const size_t count, const size_t ndims, const size_t* a_offset,
               const size_t* b_offset, const size_t* dims, const size_t* arr)
{
    const size_t a = nd_to_1d(ndims, a_offset, dims);
    const size_t b = nd_to_1d(ndims, b_offset, dims);
    return equals(count, &arr[a], &arr[b]);
}

static void
test_ndarray_equals(void)
{
    const size_t arr[] = {
        1, 1, 1, //
        1, 2, 3, //
        1, 1, 1, //
        3, 2, 1, //
    };
    const size_t ncols  = 3;
    const size_t len    = ARRAY_SIZE(arr);
    const size_t nrows  = len / ncols;
    const size_t dims[] = {ncols, nrows};
    const size_t ndims  = ARRAY_SIZE(dims);
    {
        const size_t a_offset[] = {0, 0};
        const size_t b_offset[] = {0, 1};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == false);
    }
    {
        const size_t a_offset[] = {0, 0};
        const size_t b_offset[] = {0, 2};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == true);
    }
    {
        const size_t a_offset[] = {0, 1};
        const size_t b_offset[] = {0, 3};
        ERRCHK(ndarray_equals(ncols, ndims, a_offset, b_offset, dims, arr) == false);
    }
    {
        const size_t a_offset[] = {2, 2};
        const size_t b_offset[] = {2, 3};
        ERRCHK(ndarray_equals(1, ndims, a_offset, b_offset, dims, arr) == true);
    }
}

/*
 * Matrix
 */
void
matrix_get_row(const size_t row, const size_t nrows, const size_t ncols, const size_t* mat,
               size_t* cols)
{
    ERRCHK(row < nrows);
    ncopy(ncols, &mat[row * ncols], cols);
}

static void
test_get_row(void)
{
    {
#define nrows (5)
#define ncols (2)
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {7, 8};
        size_t out[ncols];
        matrix_get_row(3, nrows, ncols, (const size_t*)in, out);
        ERRCHK(equals(ncols, row, out));
#undef nrows
#undef ncols
    }
    {
#define nrows (5)
#define ncols (2)
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {1, 2};
        size_t out[ncols];
        matrix_get_row(0, nrows, ncols, (const size_t*)in, out);
        ERRCHK(equals(ncols, row, out));
#undef nrows
#undef ncols
    }
    {
#define nrows (5)
#define ncols (2)
        const size_t in[nrows][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t row[ncols] = {9, 10};
        size_t out[ncols];
        matrix_get_row(nrows - 1, nrows, ncols, (const size_t*)in, out);
        ERRCHK(equals(ncols, row, out));
#undef nrows
#undef ncols
    }
}

void
matrix_remove_row(const size_t row, const size_t nrows, const size_t ncols, const size_t* in,
                  size_t* out)
{
    ERRCHK(row < nrows);
    ncopy(row * ncols, in, out);

    const size_t count = (nrows - row - 1) * ncols;
    if (count > 0)
        ncopy(count, &in[(row + 1) * ncols], &out[row * ncols]);
}

static void
test_remove_row(void)
{
    {
        const size_t row       = 2;
        const size_t nrows     = 5;
        const size_t ncols     = 2;
        const size_t count     = nrows * ncols;
        const size_t out_nrows = nrows - 1;
        const size_t out_count = out_nrows * ncols;

        size_t *in, *model, *out;
        nalloc(count, in);
        nalloc(out_count, model);
        nalloc(out_count, out);

        arange(1, count, in);
        arange(1, 4, model);
        arange(7, 4, &model[row * ncols]);
        matrix_remove_row(row, nrows, ncols, in, out);
        ERRCHK(equals(out_count, model, out));
        // print_array("out", out_count, out);
        // print_array("model", out_count, model);

        ndealloc(in);
        ndealloc(model);
        ndealloc(out);
    }

    {
        const size_t ncols       = 2;
        const size_t in[][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t nrows          = ARRAY_SIZE(in);
        const size_t model[][ncols] = {
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10},
        };

        const size_t out_count = (nrows - 1) * ncols;
        size_t* out;
        nalloc(out_count, out);
        matrix_remove_row(0, nrows, ncols, &in[0][0], out);
        ERRCHK(equals(out_count, &model[0][0], out));
        ndealloc(out);
    }
    {
        const size_t ncols       = 2;
        const size_t in[][ncols] = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
        };
        const size_t model[][ncols] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        const size_t nrows          = ARRAY_SIZE(in);

        const size_t out_count = (nrows - 1) * ncols;
        size_t* out;
        nalloc(out_count, out);
        matrix_remove_row(nrows - 1, nrows, ncols, &in[0][0], out);
        ERRCHK(equals(out_count, &model[0][0], out));
        ndealloc(out);
    }
}

bool
matrix_row_equals(const size_t row, const size_t nrows, const size_t ncols, const size_t* mat,
                  const size_t* cols)
{
    ERRCHK(row < nrows);
    return equals(ncols, &mat[row * ncols], cols);
}

static void
test_matrix_row_equals(void)
{
    {
        const size_t ncols        = 3;
        const size_t mat[][ncols] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const size_t nrows = ARRAY_SIZE(mat);

        ERRCHK(matrix_row_equals(0, nrows, ncols, &mat[0][0], (size_t[]){1, 2, 3}));
        ERRCHK(matrix_row_equals(1, nrows, ncols, &mat[0][0], (size_t[]){4, 5, 6}));
        ERRCHK(matrix_row_equals(2, nrows, ncols, &mat[0][0], (size_t[]){7, 8, 9}));
    }
}

static void
test_matrix(void)
{
    test_get_row();
    test_remove_row();
    test_matrix_row_equals();
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

bool
all_less_than(const size_t count, const size_t* a, const size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        if (a[i] >= b[i])
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
    test_sort();
    test_unique();
    test_repeat();
    test_unique_subsets();
    test_powzu();
    test_intersect_lines();
    test_intersect_box();
    test_any();
    test_all();
    test_within_box();
    test_next_positive_integer();
    test_nd_to_1d();
    test_ndarray_equals();
    test_to_spatial_to_linear();
    test_matrix();
}
