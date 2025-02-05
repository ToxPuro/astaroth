#pragma once
#include <iostream>

#include "ntuple.h"

namespace ac {
struct Segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    explicit Segment(const Shape& in_dims)
        : dims{in_dims}, offset{make_index(in_dims.size(), 0)}
    {
    }
    Segment(const Shape& in_dims, const Index& in_offset)
        : dims{in_dims}, offset{in_offset}
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const Segment& obj)
    {
        os << "{";
        os << "dims: " << obj.dims << ", ";
        os << "offset: " << obj.offset;
        os << "}";
        return os;
    }
};
} // namespace ac
