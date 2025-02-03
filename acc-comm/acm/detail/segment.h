#pragma once
#include <iostream>

#include "datatypes.h"

namespace ac {
struct segment {
    Shape dims;   // Dimensions of the segment
    Index offset; // Offset of the segment

    // Constructors
    explicit segment(const Shape& in_dims)
        : dims{in_dims}, offset{ac::make_vector<uint64_t>(in_dims.size(), 0)}
    {
    }
    segment(const Shape& in_dims, const Index& in_offset)
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
