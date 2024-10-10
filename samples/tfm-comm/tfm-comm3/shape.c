#include "shape.h"

#include "alloc.h"

Shape
shapeCreate(const size_t ndims)
{
    Shape shape = (Shape){
        .ndims   = ndims,
        .dims    = ac_calloc(ndims, sizeof(shape.dims[0])),
        .subdims = ac_calloc(ndims, sizeof(shape.subdims[0])),
        .offset  = ac_calloc(ndims, sizeof(shape.offset[0])),
    };
    return shape;
}

void
shapeDestroy(Shape* shape)
{
    ac_free(shape->offset);
    ac_free(shape->subdims);
    ac_free(shape->dims);
    shape->ndims   = 0;
    shape->dims    = NULL;
    shape->subdims = NULL;
    shape->offset  = NULL;
}
