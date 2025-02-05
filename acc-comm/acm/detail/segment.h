#pragma once
#include <iostream>

#include "ntuple.h"

namespace ac {
struct segment {
    ac::shape dims;   // Dimensions of the segment
    ac::index offset; // Offset of the segment

    // Constructors
    explicit segment(const ac::shape& in_dims)
        : dims{in_dims}, offset{ac::make_index(in_dims.size(), 0)}
    {
    }
    segment(const ac::shape& in_dims, const ac::index& in_offset)
        : dims{in_dims}, offset{in_offset}
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const segment& obj)
    {
        os << "{";
        os << "dims: " << obj.dims << ", ";
        os << "offset: " << obj.offset;
        os << "}";
        return os;
    }
};
} // namespace ac
