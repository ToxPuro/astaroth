#pragma once

#include "datatypes.h"

struct Segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    Segment(const Shape& in_dims)
        : dims(in_dims), offset(Index(in_dims.count))
    {
    }
    Segment(const Shape& in_dims, const Index& in_offset)
        : dims(in_dims), offset(in_offset)
    {
    }
};

__host__ std::ostream& operator<<(std::ostream& os, const Segment& obj);
