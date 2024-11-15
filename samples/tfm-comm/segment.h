#pragma once

#include "datatypes.h"

struct Segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    explicit Segment(const Shape& in_dims)
        : dims{in_dims}, offset{}
    {
    }
    Segment(const Shape& in_dims, const Index& in_offset)
        : dims{in_dims}, offset{}
    {
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const Segment& obj)
    {
        os << "{";
        os << "dims: " << obj.dims << ", ";
        os << "offset: " << obj.offset;
        os << "}";
        return os;
    }
};

__host__ std::ostream& operator<<(std::ostream& os, const Segment& obj);
