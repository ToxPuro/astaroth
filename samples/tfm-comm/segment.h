#pragma once

#include "datatypes.h"

struct Segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    Segment(const Shape& dims_)
        : dims(dims_), offset(Index(dims.count))
    {
    }
    Segment(const Shape& dims_, const Index& offset_)
        : dims(dims_), offset(offset_)
    {
    }
};

__host__ std::ostream& operator<<(std::ostream& os, const Segment& obj);
