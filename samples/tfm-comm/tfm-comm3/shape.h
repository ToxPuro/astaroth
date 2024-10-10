#pragma once
#include <stddef.h>

typedef struct {
    size_t ndims;
    size_t* dims;
    size_t* subdims;
    size_t* offset;
} Shape;

Shape shapeCreate(const size_t ndims);

void shapeDestroy(Shape* shape);
