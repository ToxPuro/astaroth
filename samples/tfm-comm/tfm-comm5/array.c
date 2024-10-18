#include "array.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "errchk.h"

/*
 * Misc
 */
void
array_set_int(const int value, const size_t count, int* arr)
{
    for (size_t i = 0; i < count; ++i)
        arr[i] = value;
}

size_t
mul_size_t(const size_t a, const size_t b)
{
    ERRCHK(b != 0);
    return a * b;
}

size_t
array_prod_size_t(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res = mul_size_t(res, arr[i]);
    return res;
}

/*
 * Primitive operations
 */
uint64_t
add_uint64_t(const uint64_t a, const uint64_t b)
{
    ERRCHK(a <= UINT64_MAX - b);
    return a + b;
}

uint64_t
sub_uint64_t(const uint64_t a, const uint64_t b)
{
    ERRCHK(a >= b);
    return a - b;
}

uint64_t
div_uint64_t(const uint64_t a, const uint64_t b)
{
    ERRCHK(b != 0);
    return a / b;
}

uint64_t
mul_uint64_t(const uint64_t a, const uint64_t b)
{
    return a * b;
}

uint64_t
int_as_uint64_t(const int value)
{
    ERRCHK(value >= 0 && (uintmax_t)value <= UINT64_MAX);
    return (uint64_t)value;
}

void
print_uint64_t(const uint64_t value)
{
    printf("%" PRIu64, value);
}

/*
 * Array operations
 */
void
array_set_uint64_t(const uint64_t value, const size_t count, uint64_t* data)
{
    for (size_t i = 0; i < count; ++i)
        data[i] = value;
}

uint64_t
array_prod_uint64_t(const size_t count, const uint64_t* arr)
{
    uint64_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res = mul_uint64_t(res, arr[i]);
    return res;
}

void
array_add_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = add_uint64_t(a[i], b[i]);
}
void
array_sub_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = sub_uint64_t(a[i], b[i]);
}
void
array_div_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = div_uint64_t(a[i], b[i]);
}
void
array_mul_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = mul_uint64_t(a[i], b[i]);
}

void
array_add_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = add_uint64_t(a[i], b);
}
void
array_sub_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = sub_uint64_t(a[i], b);
}
void
array_div_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = div_uint64_t(a[i], b);
}
void
array_mul_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = mul_uint64_t(a[i], b);
}

void
array_scal_add_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = add_uint64_t(a, b[i]);
}
void
array_scal_sub_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = sub_uint64_t(a, b[i]);
}
void
array_scal_div_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = div_uint64_t(a, b[i]);
}
void
array_scal_mul_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = mul_uint64_t(a, b[i]);
}

void
array_print_uint64_t(const size_t count, const uint64_t* data)
{
    for (size_t i = 0; i < count; ++i) {
        print_uint64_t(data[i]);
        if (i + 1 < count)
            printf(", ");
    }
}

/*
 * Nd-array operations
 */
void
ndarray_set_uint64_t(const uint64_t value, const size_t ndims, const size_t* dims,
                     const size_t* subdims, const size_t* start, uint64_t* data)
{
    if (ndims == 0) {
        *data = value;
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const size_t offset = array_prod_size_t(ndims - 1, dims);
        for (size_t i = start[ndims - 1]; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            ndarray_set_uint64_t(value, ndims - 1, dims, subdims, start, &data[i * offset]);
    }
}

/*
 * Static array
 */
StaticArray_uint64_t
make_static_array_uint64_t(const size_t count, const uint64_t* data)
{
    ERRCHK(count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t arr = {.count = count};
    memmove(arr.data, data, count * sizeof(arr.data[0]));
    return arr;
}

StaticArray_uint64_t
static_array_add_uint64_t(const StaticArray_uint64_t a, const StaticArray_uint64_t b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_add_uint64_t(a.count, a.data, b.data, c.data);
    return c;
}

StaticArray_uint64_t
static_array_sub_uint64_t(const StaticArray_uint64_t a, const StaticArray_uint64_t b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_sub_uint64_t(a.count, a.data, b.data, c.data);
    return c;
}

StaticArray_uint64_t
static_array_div_uint64_t(const StaticArray_uint64_t a, const StaticArray_uint64_t b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_div_uint64_t(a.count, a.data, b.data, c.data);
    return c;
}
StaticArray_uint64_t
static_array_mul_uint64_t(const StaticArray_uint64_t a, const StaticArray_uint64_t b)
{
    ERRCHK(a.count == b.count);
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_mul_uint64_t(a.count, a.data, b.data, c.data);
    return c;
}

StaticArray_uint64_t
static_array_add_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b)
{
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_add_scal_uint64_t(a.count, a.data, b, c.data);
    return c;
}

StaticArray_uint64_t
static_array_sub_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b)
{
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_sub_scal_uint64_t(a.count, a.data, b, c.data);
    return c;
}

StaticArray_uint64_t
static_array_div_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b)
{
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_div_scal_uint64_t(a.count, a.data, b, c.data);
    return c;
}
StaticArray_uint64_t
static_array_mul_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b)
{
    ERRCHK(a.count <= STATIC_ARRAY_MAX_COUNT_uint64_t);
    StaticArray_uint64_t c = {.count = a.count};
    array_mul_scal_uint64_t(a.count, a.data, b, c.data);
    return c;
}

void
static_array_set_uint64_t(const uint64_t value, StaticArray_uint64_t* static_array)
{
    array_set_uint64_t(value, static_array->count, static_array->data);
}

void
static_array_print_uint64_t(const StaticArray_uint64_t static_array)
{
    printf("[");
    array_print_uint64_t(static_array.count, static_array.data);
    printf("]");
}

/*
 * Dynamic array
 */

DynamicArray_StaticArray_uint64_t
dynarr_create_StaticArray_uint64_t(void)
{
    DynamicArray_StaticArray_uint64_t dynarr = {
        .length   = 0,
        .capacity = 0,
        .data     = NULL,
    };
    return dynarr;
}

void
dynarr_append_StaticArray_uint64_t(const StaticArray_uint64_t element,
                                   DynamicArray_StaticArray_uint64_t* dynarr)
{
    if (dynarr->length == dynarr->capacity) {
        dynarr->data = realloc(dynarr->data, ++dynarr->capacity * sizeof(dynarr->data[0]));
        ERRCHK(dynarr->data);
    }
    dynarr->data[dynarr->length++] = element;
}

void
dynarr_remove_StaticArray_uint64_t(const size_t index, DynamicArray_StaticArray_uint64_t* dynarr)
{
    ERRCHK(index < dynarr->length);
    const size_t count = dynarr->length - 1 - index; // Move count
    if (count > 0)
        memmove(&dynarr->data[index], &dynarr->data[index + 1], count * sizeof(dynarr->data[0]));
    --dynarr->length;
}

StaticArray_uint64_t
dynarr_get_StaticArray_uint64_t(const size_t index, const DynamicArray_StaticArray_uint64_t dynarr)
{
    ERRCHK(index < dynarr.length);
    return dynarr.data[index];
}

void
dynarr_destroy_StaticArray_uint64_t(DynamicArray_StaticArray_uint64_t* dynarr)
{
    if (dynarr->data)
        free(dynarr->data);
    dynarr->data     = NULL;
    dynarr->capacity = 0;
    dynarr->length   = 0;
}

void
dynarr_print_StaticArray_uint64_t(const DynamicArray_StaticArray_uint64_t dynarr)
{
    printf("{");
    for (size_t i = 0; i < dynarr.length; ++i) {
        static_array_print_uint64_t(dynarr.data[i]);
        if (i + 1 < dynarr.length)
            printf(", ");
    }
    printf("}");
}
