#pragma once
#include <stddef.h>
#include <stdint.h>

// Misc
void array_set_int(const int value, const size_t count, int* arr);
size_t mul_size_t(const size_t a, const size_t b);
size_t array_prod_size_t(const size_t count, const size_t* arr);

/*
 * Primitive operations
 */
uint64_t add_uint64_t(const uint64_t a, const uint64_t b);
uint64_t sub_uint64_t(const uint64_t a, const uint64_t b);
uint64_t div_uint64_t(const uint64_t a, const uint64_t b);
uint64_t mul_uint64_t(const uint64_t a, const uint64_t b);
uint64_t int_as_uint64_t(const int value);
void print_uint64_t(const uint64_t value);

/*
 * Array operations
 */
// Unary operations
void array_set_uint64_t(const uint64_t value, const size_t count, uint64_t* data);
uint64_t array_prod_uint64_t(const size_t count, const uint64_t* arr);

// Binary operations
void array_add_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);
void array_sub_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);
void array_div_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);
void array_mul_uint64_t(const size_t count, const uint64_t* a, const uint64_t* b, uint64_t* c);

void array_add_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c);
void array_sub_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c);
void array_div_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c);
void array_mul_scal_uint64_t(const size_t count, const uint64_t* a, const uint64_t b, uint64_t* c);

void array_scal_add_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c);
void array_scal_sub_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c);
void array_scal_div_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c);
void array_scal_mul_uint64_t(const uint64_t a, const size_t count, const uint64_t* b, uint64_t* c);

void array_print_uint64_t(const size_t count, const uint64_t* data);

/*
 * Nd-array operations
 */
void ndarray_set_uint64_t(const uint64_t value, const size_t ndims, const size_t* dims,
                          const size_t* subdims, const size_t* offset, uint64_t* data);

/*
 * Static array
 */
#define STATIC_ARRAY_MAX_COUNT_uint64_t ((size_t)4)
typedef struct {
    size_t count;
    uint64_t data[STATIC_ARRAY_MAX_COUNT_uint64_t];
} StaticArray_uint64_t;

StaticArray_uint64_t make_static_array_uint64_t(const size_t count, const uint64_t* data);

StaticArray_uint64_t static_array_add_uint64_t(const StaticArray_uint64_t a,
                                               const StaticArray_uint64_t b);
StaticArray_uint64_t static_array_sub_uint64_t(const StaticArray_uint64_t a,
                                               const StaticArray_uint64_t b);
StaticArray_uint64_t static_array_div_uint64_t(const StaticArray_uint64_t a,
                                               const StaticArray_uint64_t b);
StaticArray_uint64_t static_array_mul_uint64_t(const StaticArray_uint64_t a,
                                               const StaticArray_uint64_t b);

StaticArray_uint64_t static_array_add_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b);
StaticArray_uint64_t static_array_sub_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b);
StaticArray_uint64_t static_array_div_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b);
StaticArray_uint64_t static_array_mul_scal_uint64_t(const StaticArray_uint64_t a, const uint64_t b);

void static_array_set_uint64_t(const uint64_t value, StaticArray_uint64_t* static_array);

void static_array_print_uint64_t(const StaticArray_uint64_t static_array);

/*
 * Dynamic array
 */
typedef struct {
    size_t length;
    size_t capacity;
    StaticArray_uint64_t* data;
} DynamicArray_StaticArray_uint64_t;

DynamicArray_StaticArray_uint64_t dynarr_create_StaticArray_uint64_t(void);

void dynarr_append_StaticArray_uint64_t(const StaticArray_uint64_t element,
                                        DynamicArray_StaticArray_uint64_t* dynarr);

void dynarr_remove_StaticArray_uint64_t(const size_t index,
                                        DynamicArray_StaticArray_uint64_t* dynarr);

StaticArray_uint64_t
dynarr_get_StaticArray_uint64_t(const size_t index, const DynamicArray_StaticArray_uint64_t dynarr);

void dynarr_destroy_StaticArray_uint64_t(DynamicArray_StaticArray_uint64_t* dynarr);

void dynarr_print_StaticArray_uint64_t(const DynamicArray_StaticArray_uint64_t dynarr);
