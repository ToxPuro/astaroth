#pragma once

#include "datatypes.h"

namespace ac {
template <size_t N> struct segment {
    ac::shape<N> dims{};   // Dimensions of the segment
    ac::index<N> offset{}; // Offset of the segment

    // Constructors
    explicit segment(const ac::shape<N>& in_dims)
        : dims{in_dims}
    {
    }
    segment(const ac::shape<N>& in_dims, const ac::index<N>& in_offset)
        : dims{in_dims}, offset{in_offset}
    {
    }

    friend __host__ std::ostream& operator<<(std::ostream& os, const segment& obj)
    {
        os << "{";
        os << "dims: " << obj.dims << ", ";
        os << "offset: " << obj.offset;
        os << "}";
        return os;
    }
};
} // namespace ac
