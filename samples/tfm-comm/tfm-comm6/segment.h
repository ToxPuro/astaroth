#pragma once

#include "shape.h"

struct Segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    Segment(const Shape& dims) : dims(dims), offset(Index(dims.count)) {}
    Segment(const Shape& dims, const Index& offset) : dims(dims), offset(offset) {}
};

__host__ std::ostream& operator<<(std::ostream& os, const Segment& obj);
