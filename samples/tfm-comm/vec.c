#include "vec.h"

#include <stdlib.h>

#include "errchk.h"

#include "print.h"
#include <stdio.h>

typedef struct vec_s {
    size_t capacity;
    size_t len;
    const void** data;
} Vector;

Vector*
vector_create(const size_t capacity)
{
    Vector* vec = malloc(sizeof(Vector));
    ERRCHK(vec != NULL);

    vec->data = malloc(sizeof(vec->data[0]) * capacity);
    ERRCHK(vec->data != NULL);

    vec->capacity = capacity;
    vec->len      = 0;

    return vec;
}

void
vector_destroy(Vector* vec)
{
    vec->capacity = 0;
    vec->len      = 0;

    free(vec->data);
    free(vec);
}

void
vector_push(const void* ptr, Vector* vec)
{
    if (vec->len == vec->capacity) {
        vec->capacity += 1;
        vec->data = realloc(vec->data, sizeof(vec->data[0]) * vec->capacity);
        WARNING("Vector too small, reallocated");
        ERRCHK(vec->data);
    }
    vec->data[vec->len] = ptr;
    ++vec->len;
}

void*
vector_pop(Vector* vec)
{
    ERRCHK(vec->len > 0);
    --vec->len;
    return (void*)vec->data[vec->len];
}

size_t
vector_len(const Vector* vec)
{
    return vec->len;
}

void*
vector_get(const Vector* vec, const size_t i)
{
    ERRCHK(i < vec->len);
    return (void*)vec->data[i];
}

void
vector_test(void)
{
    printf("Hello\n");
    Vector* vec = vector_create(1);
    int i       = 1;
    vector_push(&i, vec);
    vector_push(&i, vec);
    vector_push(&i, vec);
    vector_push(&i, vec);
    i = 2;
    vector_pop(vec);
    print("len", vector_len(vec));
    for (size_t i = 0; i < vector_len(vec); ++i) {
        int* ptr = vector_get(vec, i);
        print("ptr", *ptr);
    }
    vector_destroy(vec);
}
