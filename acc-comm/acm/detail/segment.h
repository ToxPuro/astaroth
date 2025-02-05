#pragma once
#include <iostream>

#include "ntuple.h"

namespace ac {
struct Segment {
    ac::Shape dims;   // Dimensions of the segment
    ac::Index offset; // Offset of the segment

    // Constructors
    explicit Segment(const ac::Shape& in_dims)
        : dims{in_dims}, offset{ac::make_index(in_dims.size(), 0)}
    {
    }
    Segment(const ac::Shape& in_dims, const ac::Index& in_offset)
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
